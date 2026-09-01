"""Отрисовка витрины.

Витрина — статический HTML без единой внешней зависимости: он лежит на GitHub
Pages, а страница, которая тянет чужой CDN, однажды перестаёт открываться.
Графики сделаны двумя способами и оба выбраны осознанно:

* **Горизонтальные полосы — обычная HTML-разметка, не SVG.** У полос длинные
  русские подписи; в SVG они наезжают друг на друга при любой ширине экрана,
  а обычная сетка переносит текст сама и остаётся читаемой на телефоне.
* **Динамика по неделям — inline SVG.** Ломаная в HTML не рисуется, а тянуть
  ради одного графика библиотеку на 300 КБ — плохой обмен.

Цвета заданы токенами и продублированы под тёмную тему в двух областях: под
системную настройку и под явный выбор темы. Каждая полоса подписана числом:
это не украшение, а требование доступности — бирюзовый на светлом фоне не
добирает 3:1 контраста, и подпись обязана нести значение вместо цвета.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime

from hh_radar.text import format_count, format_money, plural_ru

# Токены палитры. Светлые и тёмные значения — не автоматическая инверсия,
# а подобранные под каждую подложку шаги одних и тех же оттенков.
LIGHT = {
    "surface": "#fcfcfb",
    "surface-2": "#f4f3f0",
    "ink": "#0b0b0b",
    "ink-2": "#52514e",
    "ink-3": "#7a7873",
    "rule": "#e3e1dc",
    "series-1": "#2a78d6",
    "series-2": "#eb6834",
    "series-3": "#1baf7a",
    "track": "#e8edf5",
    "warn-bg": "#fdf3e7",
    "warn-line": "#f0c9a0",
}
DARK = {
    "surface": "#1a1a19",
    "surface-2": "#232322",
    "ink": "#ffffff",
    "ink-2": "#c3c2b7",
    "ink-3": "#8f8e86",
    "rule": "#383835",
    "series-1": "#3987e5",
    "series-2": "#d95926",
    "series-3": "#199e70",
    "track": "#25303d",
    "warn-bg": "#2b2118",
    "warn-line": "#5a3f27",
}

#: Реэкспорт: витрина исторически была единственным потребителем
#: форматирования, и импорты из неё уже разошлись по проекту.
__all__ = [
    "BarRow",
    "TimePoint",
    "embed_json",
    "experience_label",
    "format_count",
    "format_money",
    "plural_ru",
    "render_page",
]

REPO_URL = "https://github.com/Denmurzik/hh-radar"

EXPERIENCE_LABELS = {
    "noExperience": "Без опыта",
    "between1And3": "1–3 года",
    "between3And6": "3–6 лет",
    "moreThan6": "Более 6 лет",
}


@dataclass(frozen=True, slots=True)
class BarRow:
    """Одна полоса: подпись, значение и что показать справа."""

    label: str
    value: float
    display: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class TimePoint:
    moment: datetime
    count: int


#: Ниже этого числа вакансий страница честно предупреждает, что показывает
#: не срез рынка, а демонстрационные данные. Витрина, обещающая «рынок»
#: по четырём объявлениям, — это ровно та ложь, из-за которой не верят
#: остальным цифрам.
MIN_MEANINGFUL_VACANCIES = 200


def render_page(
    *,
    generated_at: datetime,
    stats: list[tuple[str, str, str]],
    skills: list[BarRow],
    salaries: list[BarRow],
    timeline: list[TimePoint],
    employers: list[tuple[str, int]],
    examples: list[tuple[str, str]],
    period: tuple[datetime | None, datetime | None],
    payload: dict[str, object],
    vacancies_total: int,
) -> str:
    """Собрать всю страницу целиком."""
    # Секции считаются заранее: вызовы функций внутри f-строки форматтер
    # разворачивает в нечитаемую лапшу.
    period_text = _period_text(period)
    tiles = "".join(_tile(title, value, note) for title, value, note in stats)
    warning = _sample_data_warning(vacancies_total)
    skills_block = _bar_section(
        "Какие навыки требуют чаще всего",
        "Доля вакансий из базы, в которых навык указан в требованиях. "
        "Считается по нормализованным названиям: «Python» и «python» — один навык.",
        skills,
        "series-1",
        "skills",
    )
    salary_block = _bar_section(
        "Медианная зарплата по уровню опыта",
        "Медиана нижней границы вилки, приведённой к рублям. Вакансии без указанной "
        "зарплаты в расчёт не входят — их доля указана в карточке выше.",
        salaries,
        "series-2",
        "salary",
    )
    timeline_block = _timeline_section(timeline)
    employers_block = _employers_section(employers)
    examples_block = _examples_section(examples)
    raw_payload = embed_json(payload)
    description = (
        "Собственная база вакансий hh.ru с полнотекстовым и семантическим поиском. "
        "Витрина собрана программой из данных базы."
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hh-radar — срез рынка AI и автоматизации</title>
<meta name="description" content="{description}">
<style>{_css()}</style>
</head>
<body>
<div class="wrap">

  <header class="head">
    <p class="eyebrow">hh-radar · витрина</p>
    <h1>Что происходит на рынке AI и автоматизации</h1>
    <p class="lede">
      Страница собрана программой из собственной базы вакансий hh.ru — той же,
      которую LLM-агент опрашивает через MCP. Ни одна цифра здесь не вписана руками.
    </p>
    <p class="meta">{period_text} · обновлено {generated_at:%d.%m.%Y %H:%M} ·
      <a href="{REPO_URL}">исходный код</a></p>
  </header>

  {warning}

  <section class="tiles" aria-label="Ключевые числа">{tiles}
  </section>

  {skills_block}

  {salary_block}

  {timeline_block}

  {employers_block}

  {examples_block}

  <footer class="foot">
    <p><strong>Личный проект.</strong> Написан для собственного поиска работы: искал
    вакансии руками, надоело, автоматизировал. Данные получены через официальное API
    hh.ru с токеном приложения, хранятся локально и показаны здесь только агрегатами.</p>
    <p>Курсы валют для приведения зарплат — статичные и приблизительные: точность
    здесь не нужна, нужна общая шкала для сортировки.</p>
    <p><a href="{REPO_URL}">github.com/Denmurzik/hh-radar</a>
    · <a href="data.json">данные этой страницы в JSON</a></p>
  </footer>
</div>
<script id="showcase-data" type="application/json">{raw_payload}</script>
<script>{_js()}</script>
</body>
</html>
"""


# ------------------------------------------------------------------ блоки --


def embed_json(payload: object) -> str:
    """Сериализовать данные для вставки внутрь тега ``<script>``.

    ``json.dumps`` не трогает ``<`` и ``>``, а браузер закрывает блок скрипта
    на первой же последовательности ``</script>`` — где бы она ни встретилась,
    хоть внутри строкового литерала. Названия навыков и вакансий приходят от
    работодателей, то есть это чужой текст: навык с именем
    ``</script><img onerror=...>`` вырвался бы из тега и выполнился.

    Экранируем три символа их юникодными escape-последовательностями. JSON от
    этого не перестаёт быть валидным, а разорвать тег больше нечем.
    """
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _sample_data_warning(vacancies_total: int) -> str:
    """Предупреждение, если база наполнена примерами, а не настоящим сбором."""
    if vacancies_total >= MIN_MEANINGFUL_VACANCIES:
        return ""
    return (
        '<p class="warn"><strong>Это демонстрационные данные.</strong> '
        f"В базе всего {vacancies_total} "
        f"{plural_ru(vacancies_total, 'вакансия', 'вакансии', 'вакансий')} — "
        "столько кладёт команда "
        "<code>hh-radar seed</code> для знакомства с проектом. Цифры ниже "
        "посчитаны честно, но говорить по ним о рынке нельзя. Настоящий срез "
        "появляется после <code>hh-radar ingest</code> с токеном приложения hh.</p>"
    )


def _tile(title: str, value: str, note: str) -> str:
    return f"""
    <div class="tile">
      <p class="tile-title">{html.escape(title)}</p>
      <p class="tile-value">{html.escape(value)}</p>
      <p class="tile-note">{html.escape(note)}</p>
    </div>"""


def _bar_section(title: str, blurb: str, rows: list[BarRow], series: str, kind: str) -> str:
    if not rows:
        return _empty_section(title, blurb)
    top = max(row.value for row in rows) or 1.0
    bars = "".join(
        f"""
      <div class="bar-row" data-kind="{kind}" data-label="{html.escape(row.label, quote=True)}"
           data-value="{html.escape(row.display, quote=True)}"
           data-note="{html.escape(row.note or "", quote=True)}" tabindex="0">
        <div class="bar-label">{html.escape(row.label)}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width: {max(row.value / top * 100, 1.2):.1f}%;
               background: var(--{series});"></div>
        </div>
        <div class="bar-value">{html.escape(row.display)}</div>
      </div>"""
        for row in rows
    )
    return f"""
  <section class="block">
    <h2>{html.escape(title)}</h2>
    <p class="blurb">{html.escape(blurb)}</p>
    <div class="bars">{bars}</div>
  </section>"""


def _empty_section(title: str, blurb: str) -> str:
    return f"""
  <section class="block">
    <h2>{html.escape(title)}</h2>
    <p class="blurb">{html.escape(blurb)}</p>
    <p class="empty">Данных пока нет — база не наполнена.</p>
  </section>"""


def _timeline_section(points: list[TimePoint]) -> str:
    title = "Сколько вакансий публикуют по неделям"
    blurb = (
        "Количество публикаций по неделям. Провалы на краях — это границы окна сбора, "
        "а не спад на рынке: показываем как есть, без сглаживания."
    )
    if len(points) < 2:
        return _empty_section(title, blurb)

    width, height = 720, 220
    pad_left, pad_right, pad_top, pad_bottom = 44, 12, 16, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    top = max(point.count for point in points) or 1
    step = plot_w / (len(points) - 1)

    coords = [
        (pad_left + index * step, pad_top + plot_h - (point.count / top) * plot_h)
        for index, point in enumerate(points)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"{pad_left},{pad_top + plot_h} {line} {pad_left + plot_w},{pad_top + plot_h}"

    ticks = _nice_ticks(top)
    grid = "".join(
        f'<line class="grid" x1="{pad_left}" x2="{width - pad_right}" '
        f'y1="{pad_top + plot_h - (tick / top) * plot_h:.1f}" '
        f'y2="{pad_top + plot_h - (tick / top) * plot_h:.1f}" />'
        f'<text class="axis" x="{pad_left - 8}" '
        f'y="{pad_top + plot_h - (tick / top) * plot_h + 4:.1f}" text-anchor="end">{tick}</text>'
        for tick in ticks
    )

    dots = "".join(
        f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" '
        f'data-label="неделя с {point.moment:%d.%m.%Y}" data-value="{point.count}" tabindex="0" />'
        for (x, y), point in zip(coords, points, strict=True)
    )

    first, last = points[0], points[-1]
    x_labels = (
        f'<text class="axis" x="{pad_left}" y="{height - 8}">{first.moment:%d.%m}</text>'
        f'<text class="axis" x="{width - pad_right}" y="{height - 8}" '
        f'text-anchor="end">{last.moment:%d.%m}</text>'
    )

    return f"""
  <section class="block">
    <h2>{html.escape(title)}</h2>
    <p class="blurb">{html.escape(blurb)}</p>
    <div class="chart-scroll">
      <svg viewBox="0 0 {width} {height}" role="img"
           aria-label="Публикации вакансий по неделям" class="timeline">
        {grid}
        <polygon class="area" points="{area}" />
        <polyline class="line" points="{line}" />
        {dots}
        {x_labels}
      </svg>
    </div>
  </section>"""


def _employers_section(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{index}</td><td>{html.escape(name)}</td><td class='num'>{count}</td></tr>"
        for index, (name, count) in enumerate(rows, start=1)
    )
    return f"""
  <section class="block">
    <h2>Кто больше всех нанимает</h2>
    <p class="blurb">Работодатели по числу вакансий в базе за период сбора.</p>
    <div class="chart-scroll">
      <table>
        <thead><tr><th>#</th><th>Работодатель</th><th class="num">Вакансий</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
  </section>"""


def _examples_section(examples: list[tuple[str, str]]) -> str:
    if not examples:
        return ""
    items = "".join(
        f"""
      <li>
        <p class="ask">{html.escape(question)}</p>
        <p class="answer">{html.escape(answer)}</p>
      </li>"""
        for question, answer in examples
    )
    return f"""
  <section class="block">
    <h2>Что у этой базы спрашивает агент</h2>
    <p class="blurb">Те же цифры, но через MCP: инструменты подключены к Claude Desktop,
    и на вопрос в свободной форме агент отвечает выборкой из базы, а не догадкой.</p>
    <ul class="qa">{items}</ul>
  </section>"""


# ------------------------------------------------------------- оформление --


def _css() -> str:
    light = "\n    ".join(f"--{key}: {value};" for key, value in LIGHT.items())
    dark = "\n    ".join(f"--{key}: {value};" for key, value in DARK.items())
    return f"""
  :root {{
    color-scheme: light;
    {light}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      {dark}
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    {dark}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface);
    color: var(--ink);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 48px 20px 96px; }}
  .head {{ border-bottom: 1px solid var(--rule); padding-bottom: 28px; margin-bottom: 36px; }}
  .eyebrow {{
    margin: 0 0 10px; font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--ink-3); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  h1 {{ margin: 0 0 12px; font-size: clamp(28px, 5vw, 40px); line-height: 1.15; }}
  h2 {{ margin: 0 0 6px; font-size: 21px; line-height: 1.25; }}
  .lede {{ margin: 0 0 14px; font-size: 18px; color: var(--ink-2); max-width: 62ch; }}
  .meta {{ margin: 0; font-size: 14px; color: var(--ink-3); }}
  a {{ color: var(--series-1); }}
  .blurb {{ margin: 0 0 18px; font-size: 14px; color: var(--ink-2); max-width: 68ch; }}
  .block {{ margin: 0 0 48px; }}
  .empty {{ color: var(--ink-3); font-style: italic; }}
  .warn {{
    background: var(--warn-bg); border: 1px solid var(--warn-line);
    border-radius: 8px; padding: 14px 16px; margin: 0 0 28px;
    font-size: 14.5px; color: var(--ink-2); max-width: none;
  }}
  .warn strong {{ color: var(--ink); }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em;
    background: var(--surface-2); border-radius: 4px; padding: 1px 5px;
  }}

  .tiles {{
    display: grid; gap: 12px; margin-bottom: 48px;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  }}
  .tile {{
    background: var(--surface-2); border: 1px solid var(--rule);
    border-radius: 10px; padding: 16px 18px;
  }}
  .tile-title {{ margin: 0; font-size: 13px; color: var(--ink-2); }}
  .tile-value {{
    margin: 6px 0 2px; font-size: 27px; font-weight: 650; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }}
  .tile-note {{ margin: 0; font-size: 12px; color: var(--ink-3); }}

  .bars {{ display: flex; flex-direction: column; gap: 9px; }}
  .bar-row {{
    display: grid; grid-template-columns: minmax(120px, 190px) 1fr auto;
    align-items: center; gap: 14px; border-radius: 6px; outline: none;
  }}
  .bar-row:hover, .bar-row:focus-visible {{ background: var(--surface-2); }}
  .bar-label {{ font-size: 14px; color: var(--ink-2); }}
  .bar-track {{ background: var(--track); border-radius: 4px; height: 12px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .bar-value {{
    font-size: 14px; font-variant-numeric: tabular-nums; color: var(--ink);
    min-width: 68px; text-align: right;
  }}
  @media (max-width: 560px) {{
    .bar-row {{ grid-template-columns: 1fr auto; }}
    .bar-track {{ grid-column: 1 / -1; }}
  }}

  .chart-scroll {{ overflow-x: auto; }}
  svg.timeline {{ width: 100%; min-width: 520px; height: auto; display: block; }}
  .grid {{ stroke: var(--rule); stroke-width: 1; }}
  .axis {{ fill: var(--ink-3); font-size: 11px; font-family: ui-monospace, monospace; }}
  .line {{ fill: none; stroke: var(--series-1); stroke-width: 2;
           stroke-linejoin: round; stroke-linecap: round; }}
  .area {{ fill: var(--series-1); opacity: .09; }}
  .dot {{ fill: var(--series-1); stroke: var(--surface); stroke-width: 2; cursor: pointer; }}
  .dot:hover, .dot:focus-visible {{ r: 6; }}

  table {{ border-collapse: collapse; width: 100%; min-width: 420px; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); }}
  th {{ color: var(--ink-2); font-weight: 600; font-size: 13px; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  .qa {{ list-style: none; padding: 0; margin: 0; display: flex;
         flex-direction: column; gap: 16px; }}
  .qa li {{ border-left: 3px solid var(--series-3); padding-left: 14px; }}
  .ask {{ margin: 0 0 4px; font-weight: 600; }}
  .answer {{ margin: 0; color: var(--ink-2); font-size: 15px; }}

  .foot {{ border-top: 1px solid var(--rule); padding-top: 22px;
           font-size: 14px; color: var(--ink-2); }}
  .foot p {{ margin: 0 0 10px; max-width: 68ch; }}

  #tip {{
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--ink); color: var(--surface); padding: 6px 10px;
    border-radius: 6px; font-size: 13px; line-height: 1.4; max-width: 260px; z-index: 10;
  }}
  #tip.on {{ opacity: 1; }}
"""


def _js() -> str:
    """Подсказка при наведении.

    Ванильный JS на два десятка строк: тянуть ради тултипа библиотеку —
    это плюс 300 КБ и внешний CDN на странице, которая должна открываться
    всегда.
    """
    return """
(function () {
  var tip = document.createElement('div');
  tip.id = 'tip';
  document.body.appendChild(tip);

  function show(event, title, value) {
    tip.textContent = value ? title + ': ' + value : title;
    tip.classList.add('on');
    move(event);
  }
  function move(event) {
    var x = (event.clientX || 0) + 14;
    var y = (event.clientY || 0) + 16;
    var box = tip.getBoundingClientRect();
    if (x + box.width > window.innerWidth - 8) x = window.innerWidth - box.width - 8;
    if (y + box.height > window.innerHeight - 8) y = (event.clientY || 0) - box.height - 12;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hide() { tip.classList.remove('on'); }

  document.querySelectorAll('.bar-row').forEach(function (row) {
    var note = row.dataset.note;
    var text = row.dataset.label + (note ? ' — ' + note : '');
    row.addEventListener('mousemove', function (e) { show(e, text, row.dataset.value); });
    row.addEventListener('mouseleave', hide);
    row.addEventListener('focus', function (e) { show(e, text, row.dataset.value); });
    row.addEventListener('blur', hide);
  });

  document.querySelectorAll('.dot').forEach(function (dot) {
    dot.addEventListener('mousemove', function (e) {
      show(e, dot.dataset.label, dot.dataset.value + ' вакансий');
    });
    dot.addEventListener('mouseleave', hide);
    dot.addEventListener('focus', function (e) {
      show(e, dot.dataset.label, dot.dataset.value + ' вакансий');
    });
    dot.addEventListener('blur', hide);
  });
})();
"""


# --------------------------------------------------------------- мелочи ----


def _nice_ticks(top: int, count: int = 4) -> list[int]:
    """Круглые отметки по вертикали. Без них график — просто линия в воздухе."""
    if top <= 0:
        return [0]
    raw = top / count
    magnitude = 10 ** max(len(str(int(raw))) - 1, 0)
    step = max(round(raw / magnitude) * magnitude, 1)
    ticks = list(range(0, top + step, step))
    return ticks[: count + 1]


def _period_text(period: tuple[datetime | None, datetime | None]) -> str:
    start, end = period
    if start is None or end is None:
        return "период не определён"
    return f"вакансии, опубликованные {start:%d.%m.%Y} — {end:%d.%m.%Y}"


def experience_label(experience_id: str | None) -> str:
    if not experience_id:
        return "не указан"
    return EXPERIENCE_LABELS.get(experience_id, experience_id)
