"""Тесты отрисовки витрины.

Витрина — первое, что видит человек по ссылке, и единственная часть проекта,
где ошибка стоит не «неверный ответ», а «выглядит несерьёзно». Поэтому
проверяются в основном две вещи: страница не ломается на пустых данных и
не врёт про объём базы.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hh_radar.showcase.render import (
    MIN_MEANINGFUL_VACANCIES,
    BarRow,
    TimePoint,
    experience_label,
    format_count,
    format_money,
    plural_ru,
    render_page,
)


def _page(**overrides: object) -> str:
    base: dict[str, object] = {
        "generated_at": datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        "stats": [("вакансий в базе", "1 200", "работодателей: 300")],
        "skills": [BarRow("Python", 620, "52%", "620 вакансий")],
        "salaries": [BarRow("1–3 года", 150000.0, "150 000 ₽", None)],
        "timeline": [
            TimePoint(datetime(2026, 8, 3, tzinfo=UTC), 40),
            TimePoint(datetime(2026, 8, 10, tzinfo=UTC), 65),
            TimePoint(datetime(2026, 8, 17, tzinfo=UTC), 51),
        ],
        "employers": [("Глобус-М", 12)],
        "examples": [("Вопрос?", "Ответ.")],
        "period": (datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 31, tzinfo=UTC)),
        "payload": {"database": {"vacancies_total": 1200}},
        "vacancies_total": 1200,
    }
    base.update(overrides)
    return render_page(**base)  # type: ignore[arg-type]


class TestPluralRu:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, "вакансия"),
            (2, "вакансии"),
            (4, "вакансии"),
            (5, "вакансий"),
            (11, "вакансий"),
            (12, "вакансий"),
            (14, "вакансий"),
            (21, "вакансия"),
            (22, "вакансии"),
            (25, "вакансий"),
            (111, "вакансий"),
            (143, "вакансии"),
            (0, "вакансий"),
        ],
    )
    def test_forms(self, count: int, expected: str) -> None:
        assert plural_ru(count, "вакансия", "вакансии", "вакансий") == expected


class TestRenderPage:
    def test_renders_valid_looking_html(self) -> None:
        page = _page()
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
        assert '<html lang="ru">' in page

    def test_theme_tokens_defined_for_both_modes(self) -> None:
        """Тёмная тема должна выигрывать и от системной настройки, и от переключателя."""
        page = _page()
        assert "prefers-color-scheme: dark" in page
        assert ':root[data-theme="dark"]' in page
        assert ':root:not([data-theme="light"])' in page

    def test_no_external_resources(self) -> None:
        """Страница на Pages, тянущая чужой CDN, однажды перестанет открываться."""
        page = _page()
        assert "cdn." not in page
        assert "<script src=" not in page
        assert "<link rel=\"stylesheet\"" not in page

    def test_every_bar_is_labelled_with_its_value(self) -> None:
        """Требование доступности: значение несёт подпись, а не цвет полосы."""
        page = _page(skills=[BarRow("Python", 620, "52%", None)])
        assert '<div class="bar-value">52%</div>' in page

    def test_user_content_is_escaped(self) -> None:
        page = _page(employers=[("<script>alert(1)</script>", 3)])
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_survives_completely_empty_database(self) -> None:
        page = _page(
            stats=[],
            skills=[],
            salaries=[],
            timeline=[],
            employers=[],
            examples=[],
            period=(None, None),
            vacancies_total=0,
        )
        assert "Данных пока нет" in page
        assert "период не определён" in page

    def test_timeline_needs_two_points(self) -> None:
        page = _page(timeline=[TimePoint(datetime(2026, 8, 3, tzinfo=UTC), 40)])
        assert "Данных пока нет" in page


class TestSampleDataWarning:
    def test_warns_when_the_database_is_tiny(self) -> None:
        page = _page(vacancies_total=4)
        assert "демонстрационные данные" in page
        assert "4 вакансии" in page

    def test_no_warning_on_a_real_dataset(self) -> None:
        page = _page(vacancies_total=MIN_MEANINGFUL_VACANCIES)
        assert "демонстрационные данные" not in page


class TestFormatting:
    def test_money_uses_thin_grouping_and_ruble_sign(self) -> None:
        assert format_money(150000) == "150 000 ₽"

    def test_money_none_is_a_dash_not_zero(self) -> None:
        assert format_money(None) == "—"

    def test_count_grouping(self) -> None:
        assert format_count(1200) == "1 200"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("noExperience", "Без опыта"),
            ("between1And3", "1–3 года"),
            (None, "не указан"),
            ("somethingNew", "somethingNew"),
        ],
    )
    def test_experience_labels(self, raw: str | None, expected: str) -> None:
        assert experience_label(raw) == expected
