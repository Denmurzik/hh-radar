"""MCP-сервер hh-radar поверх слоя запросов и профиля кандидата.

Каждый инструмент открывает свою короткоживущую сессию через
``session_scope()`` и отдаёт наружу обычные JSON-совместимые dict/list — без
ORM-объектов и без ``datetime`` (только ISO-строки через ``to_dict()``
dataclass'ов из ``hh_radar.db.queries``). Ни один инструмент не должен ронять
процесс сервера: любая ошибка внутри превращается в
``{"error": "...", "hint": "..."}`` на русском, а не в трейсбек.

Инструменты объявлены как обычные функции модульного уровня (а не только
внутри декоратора) по двум причинам: их можно звать напрямую в тестах через
monkeypatch, и их регистрация в MCPServer вынесена в ``build_server()`` —
фабрику, на которую опирается CI-смоук-тест (``python -c "from
hh_radar.mcp_server.server import build_server; print(build_server().name)"``).
Ни импорт модуля, ни сама фабрика не обращаются к базе и не грузят fastembed —
``semantic_search`` импортирует ``hh_radar.rag`` лениво, внутри своего тела.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from hh_radar.db import queries
from hh_radar.db.queries import MAX_LIMIT
from hh_radar.db.session import ping, session_scope
from hh_radar.profile import ProfileFormatError, ProfileNotFoundError, load_profile, match_vacancy

logger = logging.getLogger(__name__)

#: После этой длины описание вакансии обрезается в выдаче поиска — полный
#: текст отдаёт только get_vacancy. Число выбрано с запасом: этого хватает,
#: чтобы понять суть вакансии, не раздувая контекст модели на каждый хит.
_SEARCH_SNIPPET_CHARS = 400


def _db_unavailable() -> dict[str, Any]:
    return {
        "error": "База данных недоступна.",
        "hint": "Поднимите её командой: docker compose up -d db",
    }


def _clamp(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


def _truncate(text: str | None, limit: int = _SEARCH_SNIPPET_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def search_vacancies(
    query: str = "",
    area_id: int | None = None,
    salary_min_rub: int | None = None,
    experience: str | None = None,
    remote_only: bool = False,
    published_within_days: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            results = queries.search_vacancies(
                session,
                query,
                area_id=area_id,
                salary_min_rub=salary_min_rub,
                experience_ids=[experience] if experience else None,
                remote_only=remote_only,
                published_within_days=published_within_days,
                limit=_clamp(limit),
            )
        items = []
        for item in results:
            payload = item.to_dict()
            payload["description"] = _truncate(payload["description"])
            items.append(payload)
        return {"results": items, "count": len(items)}
    except Exception as exc:
        logger.exception("search_vacancies упал")
        return {"error": "Не удалось выполнить поиск вакансий.", "hint": str(exc)}


def get_vacancy(vacancy_id: int) -> dict[str, Any]:
    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            detail = queries.get_vacancy(session, vacancy_id)
        if detail is None:
            return {
                "error": f"Вакансия с id={vacancy_id} не найдена в базе.",
                "hint": "Проверьте id или найдите вакансию через search_vacancies.",
            }
        return detail.to_dict()
    except Exception as exc:
        logger.exception("get_vacancy упал")
        return {"error": "Не удалось загрузить карточку вакансии.", "hint": str(exc)}


def skill_stats(
    query: str | None = None,
    published_within_days: int | None = None,
    area_id: int | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            stats = queries.skill_stats(
                session,
                query=query,
                published_within_days=published_within_days,
                area_id=area_id,
                top_n=_clamp(top_n),
            )
        return {"skills": [s.to_dict() for s in stats], "count": len(stats)}
    except Exception as exc:
        logger.exception("skill_stats упал")
        return {"error": "Не удалось посчитать статистику по навыкам.", "hint": str(exc)}


def market_overview(
    query: str | None = None,
    published_within_days: int | None = None,
    area_id: int | None = None,
) -> dict[str, Any]:
    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            overview = queries.market_overview(
                session,
                query=query,
                published_within_days=published_within_days,
                area_id=area_id,
            )
        return overview.to_dict()
    except Exception as exc:
        logger.exception("market_overview упал")
        return {"error": "Не удалось посчитать срез рынка.", "hint": str(exc)}


def compare_to_profile(vacancy_id: int, profile_path: str | None = None) -> dict[str, Any]:
    try:
        profile = load_profile(Path(profile_path) if profile_path else None)
    except ProfileNotFoundError as exc:
        return {"error": "Профиль кандидата не найден.", "hint": str(exc)}
    except ProfileFormatError as exc:
        return {"error": "Профиль кандидата повреждён.", "hint": str(exc)}

    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            detail = queries.get_vacancy(session, vacancy_id)
        if detail is None:
            return {
                "error": f"Вакансия с id={vacancy_id} не найдена в базе.",
                "hint": "Проверьте id или найдите вакансию через search_vacancies.",
            }
        return match_vacancy(profile, detail).to_dict()
    except Exception as exc:
        logger.exception("compare_to_profile упал")
        return {"error": "Не удалось сравнить вакансию с профилем.", "hint": str(exc)}


def semantic_search(query: str, limit: int = 10, min_similarity: float = 0.0) -> dict[str, Any]:
    try:
        from hh_radar.rag.search import semantic_search as _semantic_search
    except ImportError:
        return {
            "error": "Семантический поиск недоступен.",
            "hint": "Установите extra rag: uv pip install -e .[rag]",
        }

    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            hits = _semantic_search(
                session, query, limit=_clamp(limit), min_similarity=min_similarity
            )
        return {"results": [h.to_dict() for h in hits], "count": len(hits)}
    except Exception as exc:
        logger.exception("semantic_search упал")
        return {"error": "Семантический поиск завершился ошибкой.", "hint": str(exc)}


def db_status() -> dict[str, Any]:
    if not ping():
        return _db_unavailable()
    try:
        with session_scope() as session:
            status = queries.db_status(session)
        return status.to_dict()
    except Exception as exc:
        logger.exception("db_status упал")
        return {"error": "Не удалось получить статус базы.", "hint": str(exc)}


def build_server() -> MCPServer:
    """Собирает MCPServer со всеми инструментами hh-radar.

    Ничего не открывает и никуда не подключается: конструирование сервера —
    чистая регистрация метаданных инструментов. Это то, что вызывает
    CI-смоук-тест без живой базы и без скачанной модели эмбеддингов.
    """
    srv = MCPServer(
        name="hh-radar",
        instructions=(
            "Инструменты для работы с локальной базой вакансий hh.ru: полнотекстовый "
            "и семантический поиск, карточка вакансии, агрегаты по рынку и навыкам, "
            "сравнение вакансии с профилем кандидата. Начинайте с db_status, чтобы "
            "понимать, какие данные вообще есть и за какой период — иначе легко "
            "выдумать вакансию или статистику, которых в базе нет. Для поиска по "
            "конкретным словам используйте search_vacancies, для поиска по смыслу — "
            "semantic_search."
        ),
    )

    srv.tool(
        description=(
            "Полнотекстовый поиск вакансий по словам в названии и описании "
            "(Postgres tsvector, синтаксис websearch: кавычки для точной фразы, "
            "минус перед словом — исключить его). Используйте, когда запрос содержит "
            "конкретные термины: должность, технологию, компанию. Для поиска по смыслу "
            "и синонимам без точных слов используйте semantic_search. Пустой query — "
            "не ошибка, вернутся вакансии по фильтрам, отсортированные по дате "
            "публикации. Описание в каждом результате обрезано; полный текст — через "
            "get_vacancy по id."
        )
    )(search_vacancies)

    srv.tool(
        description=(
            "Полная карточка одной вакансии по её id из hh.ru: незакрытое описание, "
            "работодатель, все требуемые навыки, ссылка на hh.ru. Используйте после "
            "search_vacancies/semantic_search, чтобы изучить конкретную вакансию "
            "подробно, или перед compare_to_profile."
        )
    )(get_vacancy)

    srv.tool(
        description=(
            "Статистика по навыкам: какие технологии чаще всего требуются в "
            "подходящих под фильтр вакансиях, их доля и медианная зарплата вакансий, "
            "где навык нужен. Используйте для вопросов «что сейчас востребовано» или "
            "«что учить». Не возвращает список вакансий — для него используйте "
            "search_vacancies."
        )
    )(skill_stats)

    srv.tool(
        description=(
            "Общий срез рынка по фильтру: сколько вакансий, у скольких указана "
            "зарплата и её персентили (p25/p50/p75), доля удалённых, разбивка по "
            "требуемому опыту и топ-10 работодателей. Используйте для агрегированных "
            "вопросов вида «какие сейчас зарплаты» или «много ли удалёнки» — не для "
            "списка конкретных вакансий."
        )
    )(market_overview)

    srv.tool(
        description=(
            "Честно сопоставляет вакансию с профилем кандидата (profile.yaml): "
            "verdict good/stretch/no, совпавшие и недостающие навыки, а если verdict "
            "«no» — конкретные причины отказа (blockers): зарплата ниже минимума, "
            "требуемый опыт выше принимаемого, не подходящая занятость, не удалённая "
            "работа, стоп-слово из red_flags. Используйте перед тем, как рекомендовать "
            "вакансию кандидату как хороший вариант — не оценивайте пригодность "
            "вакансии самостоятельно. По умолчанию грузит profile.yaml из корня "
            "проекта (шаблон — profile.example.yaml); profile_path позволяет указать "
            "другой файл."
        )
    )(compare_to_profile)

    srv.tool(
        description=(
            "Поиск вакансий по смыслу через векторные эмбеддинги описаний, а не по "
            "точным словам. Используйте, когда запрос описывает задачу своими словами "
            "или ищет по концепции без гарантии, что в вакансии есть именно эти слова. "
            "Для поиска по конкретным терминам используйте search_vacancies — он "
            "быстрее и не требует установленного extra `rag`. Возвращает вакансии с "
            "оценкой похожести (similarity)."
        )
    )(semantic_search)

    srv.tool(
        description=(
            "Служебная информация о состоянии базы: сколько вакансий/работодателей/"
            "навыков загружено, сколько описаний уже проиндексировано эмбеддингами, "
            "какой моделью они посчитаны, за какой период данные и когда их в "
            "последний раз собирали. Вызывайте в начале работы, чтобы понимать "
            "границы доступных данных и не выдумывать вакансии или статистику, "
            "которых в базе нет. Параметров не принимает."
        )
    )(db_status)

    return srv


#: Модульный синглтон для удобства (CLI, __main__.py) — конструирование дёшево,
#: поэтому его можно себе позволить прямо на импорте модуля.
server = build_server()
