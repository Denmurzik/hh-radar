"""Командная строка hh-radar.

Одна точка входа на все операции: собрать, дозагрузить, проиндексировать,
померить качество поиска, собрать витрину, запустить MCP-сервер.

Тяжёлые модули (эмбеддинги, витрина) импортируются внутри команд, а не
наверху файла: ``hh-radar --help`` не должен тянуть onnxruntime.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hh_radar import __version__

app = typer.Typer(
    name="hh-radar",
    help="База вакансий hh.ru с полнотекстовым и семантическим поиском.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

#: Запросы по умолчанию — домен, ради которого проект и написан.
#: Список осознанно широкий: смежные формулировки ловят вакансии,
#: которые под точным «AI-инженер» не находятся.
DEFAULT_QUERIES = [
    "AI инженер",
    "AI разработчик",
    "автоматизация бизнес-процессов",
    "n8n",
    "MCP сервер",
    "LLM интеграция",
    "промпт инженер",
    "Python автоматизация",
    "чат-бот разработчик",
    "интеграция API автоматизация",
]


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Подробный лог.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def version() -> None:
    """Версия пакета."""
    console.print(f"hh-radar {__version__}")


# ------------------------------------------------------------------ данные --


@app.command()
def ingest(
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Поисковый запрос. Можно повторять."),
    ] = None,
    days: Annotated[int, typer.Option(help="Глубина сбора в днях.")] = 30,
    area: Annotated[
        str | None, typer.Option(help="Регион hh (113 — вся Россия, 1 — Москва).")
    ] = None,
    details: Annotated[
        bool, typer.Option("--details/--no-details", help="Дозагружать полные карточки.")
    ] = True,
    detail_limit: Annotated[
        int | None, typer.Option(help="Сколько карточек дозагрузить за прогон.")
    ] = None,
) -> None:
    """Собрать вакансии с hh и разложить по таблицам."""
    from hh_radar.db.session import session_scope
    from hh_radar.hh.auth import HHAuthError
    from hh_radar.hh.client import HHClient
    from hh_radar.ingest.pipeline import ingest as run_ingest

    queries = query or DEFAULT_QUERIES
    _require_db()

    try:
        with HHClient() as client, session_scope() as session:
            report = run_ingest(
                session,
                client,
                queries=queries,
                days=days,
                area=area,
                fetch_details=details,
                detail_limit=detail_limit,
            )
    except HHAuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    for line in report.as_lines():
        console.print(line)
    if report.errors:
        console.print(f"[yellow]ошибок при дозагрузке: {len(report.errors)}[/yellow]")


@app.command()
def details(
    limit: Annotated[int | None, typer.Option(help="Сколько карточек дозагрузить.")] = None,
) -> None:
    """Дозагрузить полные карточки для вакансий, у которых их ещё нет."""
    from hh_radar.db.session import session_scope
    from hh_radar.hh.client import HHClient
    from hh_radar.ingest.pipeline import fetch_missing_details

    _require_db()
    with HHClient() as client, session_scope() as session:
        report = fetch_missing_details(session, client, limit=limit)
    for line in report.as_lines():
        console.print(line)


@app.command()
def seed() -> None:
    """Наполнить базу примерами из samples/ — без токена и без сети.

    Быстрый способ посмотреть на работающий поиск, если заводить приложение
    на dev.hh.ru ради знакомства не хочется.
    """
    from hh_radar.db.session import session_scope
    from hh_radar.ingest.seed import seed_from_samples

    _require_db()
    with session_scope() as session:
        report = seed_from_samples(session)
    console.print(f"загружено вакансий: {report.seen}, связей с навыками: {report.skills_linked}")


@app.command()
def status() -> None:
    """Что лежит в базе."""
    from hh_radar.db.queries import db_status
    from hh_radar.db.session import session_scope

    _require_db()
    with session_scope() as session:
        info = db_status(session)

    table = Table(show_header=False, box=None)
    table.add_row("вакансий", f"{info.vacancies_total:,}".replace(",", " "))
    table.add_row("работодателей", f"{info.employers_total:,}".replace(",", " "))
    table.add_row("навыков", f"{info.skills_total:,}".replace(",", " "))
    table.add_row(
        "чанков (с векторами)",
        f"{info.chunks_total:,}".replace(",", " ") + f" ({info.chunks_embedded})",
    )
    if info.published_from and info.published_to:
        table.add_row(
            "период публикаций",
            f"{info.published_from:%Y-%m-%d} — {info.published_to:%Y-%m-%d}",
        )
    console.print(table)


# ------------------------------------------------------------------- поиск --


@app.command()
def index(
    batch_size: Annotated[int, typer.Option(help="Размер батча эмбеддингов.")] = 64,
    limit: Annotated[int | None, typer.Option(help="Ограничить число вакансий.")] = None,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Пересчитать даже уже проиндексированные.")
    ] = False,
) -> None:
    """Посчитать эмбеддинги описаний и сложить их в pgvector."""
    from hh_radar.db.session import session_scope
    from hh_radar.rag.indexer import index_vacancies

    _require_db()
    with session_scope() as session:
        report = index_vacancies(
            session, batch_size=batch_size, only_missing=not rebuild, limit=limit
        )
    console.print(
        f"вакансий обработано: {report.vacancies_processed}, "
        f"чанков записано: {report.chunks_written}, "
        f"пропущено: {report.chunks_skipped}, "
        f"модель: {report.model_name}, время: {report.elapsed_seconds:.1f} с"
    )


@app.command()
def evaluate(
    queries_path: Annotated[
        str, typer.Option(help="Файл с размеченными запросами.")
    ] = "eval/queries.yaml",
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Куда записать markdown-отчёт.")
    ] = None,
) -> None:
    """Померить качество поиска: recall@k, precision@k, MRR по трём методам."""
    from pathlib import Path

    from hh_radar.db.session import session_scope
    from hh_radar.rag.evaluate import evaluate_retrieval, render_report

    _require_db()
    with session_scope() as session:
        report = evaluate_retrieval(session, Path(queries_path))

    rendered = render_report(report)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"отчёт записан: {output}")
    else:
        console.print(rendered)


@app.command()
def explain(
    query: Annotated[str, typer.Argument(help="Что искать.")] = "python автоматизация",
) -> None:
    """Показать план полнотекстового запроса.

    Существует ради одной цели: проверить, что GIN-индекс действительно
    работает, а не остаётся украшением схемы. Важная оговорка, которую
    команда печатает сама: на маленькой таблице Postgres выберет
    последовательное чтение, и это правильное решение планировщика,
    а не поломка индекса.
    """
    from sqlalchemy import func, select, text

    from hh_radar.db.models import Vacancy
    from hh_radar.db.session import session_scope
    from hh_radar.text import plural_ru

    _require_db()
    sql = text(
        "EXPLAIN ANALYZE "
        "SELECT id, name FROM vacancies "
        "WHERE search_vector @@ websearch_to_tsquery('russian', :q) "
        "ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('russian', :q)) DESC "
        "LIMIT 20"
    )
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(Vacancy)) or 0
        plan = [row[0] for row in session.execute(sql, {"q": query})]

    for line in plan:
        console.print(line)

    console.print("")
    joined = " ".join(plan)
    rows_word = plural_ru(total, "строка", "строки", "строк")
    if "Bitmap Index Scan" in joined or "Index Scan" in joined:
        console.print("[green]Индекс используется.[/green]")
    elif total < 1000:
        console.print(
            "[yellow]Планировщик выбрал последовательное чтение — и правильно "
            f"сделал: в таблице всего {total} {rows_word}, прочитать их целиком "
            "дешевле, чем идти через индекс.[/yellow]"
        )
        console.print(
            "Наберите базу командой [bold]hh-radar ingest[/bold] и повторите — "
            "на нескольких тысячах вакансий появится Bitmap Index Scan."
        )
    else:
        console.print(
            "[red]Индекс не использован при заметном объёме данных — это стоит "
            "разобрать: проверьте, что ix_vacancies_search_vector существует "
            "и что статистика собрана (ANALYZE vacancies).[/red]"
        )


# ---------------------------------------------------------------- витрина ---


@app.command()
def showcase(
    output: Annotated[str, typer.Option("--output", "-o", help="Каталог витрины.")] = "docs",
) -> None:
    """Собрать статическую витрину из данных базы (для GitHub Pages)."""
    from pathlib import Path

    from hh_radar.db.session import session_scope
    from hh_radar.showcase.build import build_showcase

    _require_db()
    with session_scope() as session:
        written = build_showcase(session, Path(output))
    for path in written:
        console.print(f"записано: {path}")


@app.command(name="mcp-config")
def mcp_config(
    write: Annotated[
        bool, typer.Option("--write", help="Вписать в конфиг Claude Desktop, а не печатать.")
    ] = False,
    path: Annotated[
        str | None, typer.Option(help="Нестандартный путь к claude_desktop_config.json.")
    ] = None,
) -> None:
    """Показать (или вписать) настройку для Claude Desktop.

    Без флага только печатает — редактировать чужой конфиг без явного
    разрешения нельзя. С флагом сохраняет резервную копию и не трогает
    остальные серверы в файле.
    """
    from pathlib import Path

    from hh_radar.mcp_server.desktop import config_path, install, render_snippet

    target = Path(path) if path else config_path()

    if not write:
        console.print(f"[dim]Файл конфигурации: {target}[/dim]")
        console.print(render_snippet())
        console.print("")
        console.print("[dim]Вписать автоматически: hh-radar mcp-config --write[/dim]")
        return

    try:
        result = install(target)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=4) from exc

    console.print(f"записано: {result.path}")
    if result.backup:
        console.print(f"резервная копия: {result.backup}")
    if result.already_present:
        console.print("[yellow]запись hh-radar уже была — обновлена[/yellow]")
    console.print("Перезапустите Claude Desktop, чтобы он подхватил сервер.")


@app.command()
def serve() -> None:
    """Запустить MCP-сервер на stdio (так его подключает Claude Desktop)."""
    from hh_radar.mcp_server.server import server

    server.run("stdio")


# -------------------------------------------------------------- internals ---


def _require_db() -> None:
    """Проверить, что база отвечает, и объяснить, что делать, если нет."""
    from hh_radar.db.session import ping

    if ping():
        return
    console.print(
        "[red]База недоступна.[/red] Поднимите её командой:\n"
        "  docker compose up -d db\n"
        "и примените миграции:\n"
        "  alembic upgrade head"
    )
    raise typer.Exit(code=3)


if __name__ == "__main__":  # pragma: no cover
    app()
