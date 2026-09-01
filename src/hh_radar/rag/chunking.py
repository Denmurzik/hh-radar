"""Разбор HTML-описаний вакансий и нарезка текста на чанки для эмбеддинга.

hh.ru отдаёт описание вакансии как HTML-фрагмент (``<p>``, ``<ul>/<li>``,
``<strong>``, ``<br/>``, html-энтити). Эмбеддинг-модель работает с обычным
текстом, поэтому сначала HTML превращается в читаемый plain-текст
(``strip_html``), затем текст режется на куски заданного размера с
перекрытием (``chunk_text``), а куски снабжаются названием вакансии
(``chunk_vacancy``).
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

#: Теги, которые в описании вакансии обозначают начало нового блока: абзац,
#: элемент списка, разрыв строки, заголовок, ячейка таблицы. При встрече
#: такого тега в текстовый поток добавляется перевод строки — иначе
#: "Требования</p><p>Опыт" слипнется в одну строку "ТребованияОпыт".
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "blockquote",
        "section",
        "hr",
    }
)


class _DescriptionParser(HTMLParser):
    """Вытаскивает текст из HTML-описания вакансии, сохраняя разбиение на блоки."""

    def __init__(self) -> None:
        # convert_charrefs=True (значение по умолчанию) сам разворачивает
        # числовые и именованные ссылки (&nbsp;, &amp;...) внутри handle_data.
        # html.unescape() ниже, в strip_html, оставлен как страховка на случай
        # сущностей, не разобранных парсером (например, в редких edge-кейсах
        # с атрибутами или самопальным HTML от работодателя).
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def _handle_block(self, tag: str) -> None:
        if tag == "li":
            self._parts.append("\n— ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_block(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # самозакрывающийся вариант, например <br/>
        self._handle_block(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_TRAILING_SPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def strip_html(raw: str) -> str:
    """Превращает HTML-описание вакансии в читаемый plain-текст.

    Блочные теги (``<p>``, ``<div>``, ``<br>``, заголовки, таблицы...)
    превращаются в перевод строки, ``<li>`` — в маркер списка "— ". HTML-энтити
    (``&nbsp;``, ``&amp;`` и т.п.) разворачиваются в символы. Несколько подряд
    идущих пустых строк схлопываются в одну.
    """
    if not raw or not raw.strip():
        return ""

    parser = _DescriptionParser()
    parser.feed(raw)
    parser.close()
    text = unescape(parser.get_text())

    # &nbsp; после unescape превращается в неразрывный пробел \xa0 — для
    # текста, который дальше режется по границам предложений, это обычный
    # пробел.
    text = text.replace("\xa0", " ")
    text = _TRAILING_SPACE_BEFORE_NEWLINE_RE.sub("\n", text)
    text = _INLINE_SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Делит текст на предложения, уважая границы абзацев и строк списка.

    Предложение никогда не "перепрыгивает" границу абзаца или строки — каждая
    строка (в том числе каждый "— " пункт списка) обрабатывается отдельно, а
    внутри неё уже ищутся концы предложений (. ! ? …).
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(line):
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)
    return units


def _split_long_sentence(sentence: str, chunk_chars: int) -> list[str]:
    """Режет предложение длиннее лимита по словам (крайний случай: например,
    длинная ссылка или перечисление без знаков препинания). Слова никогда не
    рвутся — если единственное слово само длиннее лимита, оно остаётся целым.
    """
    words = sentence.split(" ")
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > chunk_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _sentences_len(sentences: list[str]) -> int:
    return sum(len(s) for s in sentences) + max(len(sentences) - 1, 0)


def _take_tail(sentences: list[str], overlap_chars: int) -> list[str]:
    """Возвращает предложения с конца списка суммарной длиной ~overlap_chars.

    Это и есть перекрытие: конец только что закрытого чанка повторяется в
    начале следующего.
    """
    if overlap_chars <= 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total >= overlap_chars:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
    return tail


def chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Режет текст на куски не длиннее ``chunk_chars`` по границам предложений.

    Куски перекрываются: конец одного чанка (~``overlap_chars`` символов)
    повторяется в начале следующего. Это защита от того, что важная мысль
    окажется ровно на границе двух чанков и не найдётся целиком ни в одном —
    без перекрытия у эмбеддинга обеих половин была бы только часть смысла.

    Резка идёт по предложениям и абзацам, а не "по счётчику символов": слово
    никогда не разрезается пополам. Пустые и состоящие из пробелов чанки не
    возвращаются. Последний чанк может быть короче остальных — это нормально.
    """
    text = text.strip()
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunk_sentences: list[list[str]] = []
    current: list[str] = []

    for sentence in sentences:
        if len(sentence) > chunk_chars:
            if current:
                chunk_sentences.append(current)
                current = []
            for piece in _split_long_sentence(sentence, chunk_chars):
                chunk_sentences.append([piece])
            continue

        if current and _sentences_len([*current, sentence]) > chunk_chars:
            chunk_sentences.append(current)
            current = _take_tail(current, overlap_chars)

        current.append(sentence)

    if current:
        chunk_sentences.append(current)

    chunks = [" ".join(sents).strip() for sents in chunk_sentences]
    return [c for c in chunks if c.strip()]


def chunk_vacancy(
    name: str,
    description_html: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Готовит текст вакансии к эмбеддингу: HTML -> текст -> чанки с названием.

    Каждый чанк начинается с названия вакансии. Это не косметика: модель
    эмбеддинга кодирует каждый чанк независимо от остальных, и оторванный от
    контекста кусок текста — например, "требуется опыт от года и знание
    Docker" — не найдётся по запросу "python-разработчик", потому что в самом
    куске текста слова "python" просто нет, хотя речь именно о такой
    вакансии. Название в начале каждого чанка чинит эту потерю контекста
    ценой небольшого расхода бюджета символов чанка.
    """
    text = strip_html(description_html)
    if not text:
        return []

    prefix = f"{name}\n"
    # бюджет тела чанка уменьшаем на длину префикса, чтобы итоговый чанк
    # (префикс + тело) не вылезал далеко за chunk_chars
    body_budget = max(chunk_chars - len(prefix), 1)
    bodies = chunk_text(text, chunk_chars=body_budget, overlap_chars=overlap_chars)
    return [f"{prefix}{body}" for body in bodies]
