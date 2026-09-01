"""Тесты нарезки текста: strip_html, chunk_text, chunk_vacancy.

Без базы данных и без загрузки эмбеддинг-модели — маркеры integration/embeddings
не нужны.
"""

from __future__ import annotations

from hh_radar.rag.chunking import chunk_text, chunk_vacancy, strip_html

# Реалистичный фрагмент описания вакансии в стиле hh.ru: вложенные ul/li,
# &nbsp;, &amp;, самозакрывающийся <br/>.
_HH_LIKE_HTML = """
<p>Мы ищем <strong>Python-разработчика</strong>&nbsp;в команду автоматизации.</p>
<p>Обязанности:</p>
<ul>
  <li>Разработка интеграций с CRM
    <ul>
      <li>Работа с REST API</li>
      <li>Обработка webhook&#39;ов</li>
    </ul>
  </li>
  <li>Поддержка&nbsp;существующих pipeline</li>
  <li>Работа с очередями &amp; фоновыми задачами</li>
</ul>
<p>Будет плюсом:<br/>опыт с Docker и Kubernetes.</p>
"""


def test_strip_html_removes_tags_and_keeps_readable_text() -> None:
    text = strip_html(_HH_LIKE_HTML)
    assert "<" not in text
    assert ">" not in text
    assert "Python-разработчика" in text
    assert "команду автоматизации" in text


def test_strip_html_unescapes_entities() -> None:
    text = strip_html(_HH_LIKE_HTML)
    # &amp; развёрнут в "&", а не остался как сущность
    assert "очередями & фоновыми" in text
    # &nbsp; свёрнут в обычный пробел, а не остался невидимым \xa0
    assert "\xa0" not in text
    assert "Поддержка существующих" in text
    # &#39; -> апостроф
    assert "webhook'ов" in text


def test_strip_html_converts_li_to_markers() -> None:
    text = strip_html(_HH_LIKE_HTML)
    assert "— Разработка интеграций с CRM" in text
    assert "— Работа с REST API" in text
    assert "— Работа с очередями" in text


def test_strip_html_collapses_blank_lines() -> None:
    text = strip_html("<p>Первый</p><p></p><p></p><p>Второй</p>")
    assert "\n\n\n" not in text


def test_strip_html_empty_input() -> None:
    assert strip_html("") == ""
    assert strip_html("   \n\t  ") == ""


def test_chunk_text_empty_returns_empty_list() -> None:
    assert chunk_text("", chunk_chars=500, overlap_chars=50) == []
    assert chunk_text("   ", chunk_chars=500, overlap_chars=50) == []


def test_chunk_text_short_text_is_one_chunk() -> None:
    text = "Короткое описание вакансии в одно предложение."
    chunks = chunk_text(text, chunk_chars=900, overlap_chars=150)
    assert chunks == [text]


def test_chunk_text_no_empty_or_whitespace_chunks() -> None:
    text = " ".join(f"Предложение номер {i} с текстом для нарезки." for i in range(1, 20))
    chunks = chunk_text(text, chunk_chars=150, overlap_chars=30)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_respects_chunk_chars_budget() -> None:
    text = " ".join(f"Предложение номер {i} с текстом для нарезки." for i in range(1, 20))
    chunk_chars = 150
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=30)
    # отдельные предложения короче лимита, поэтому ни один чанк не должен
    # сильно превышать бюджет (небольшой запас на округление по предложениям)
    assert all(len(c) <= chunk_chars + 60 for c in chunks)


def test_chunk_text_overlap_actually_overlaps() -> None:
    text = " ".join(
        f"Предложение номер {i} с некоторым текстом для проверки." for i in range(1, 15)
    )
    chunks = chunk_text(text, chunk_chars=200, overlap_chars=50)
    assert len(chunks) >= 2

    def suffix_prefix_overlap(a: str, b: str) -> int:
        for length in range(min(len(a), len(b)), 0, -1):
            if a[-length:] == b[:length]:
                return length
        return 0

    overlaps = [suffix_prefix_overlap(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    assert all(o > 0 for o in overlaps), "между соседними чанками должно быть перекрытие"


def test_chunk_text_never_splits_a_word() -> None:
    # длинное "предложение" без знаков препинания и с редкими пробелами —
    # проверяем, что порезка идёт по словам, а не разрезает слово пополам
    long_word = "а" * 500
    text = f"Начало текста. {long_word} конец текста."
    chunks = chunk_text(text, chunk_chars=100, overlap_chars=10)
    joined = "".join(chunks)
    assert long_word in joined


def test_chunk_vacancy_each_chunk_starts_with_vacancy_name() -> None:
    text = " ".join(f"Требование номер {i}: знание технологии и опыт работы." for i in range(1, 20))
    html = f"<p>{text}</p>"
    name = "Python-разработчик (удалённо)"
    chunks = chunk_vacancy(name, html, chunk_chars=200, overlap_chars=30)
    assert len(chunks) > 1
    assert all(c.startswith(f"{name}\n") for c in chunks)


def test_chunk_vacancy_empty_description_returns_empty_list() -> None:
    assert chunk_vacancy("Вакансия без описания", "", chunk_chars=900, overlap_chars=150) == []
    assert chunk_vacancy("Вакансия без описания", "   ", chunk_chars=900, overlap_chars=150) == []
