"""Оценка качества ретривала: recall@k, precision@k, MRR по трём методам поиска.

Честный раздел про то, где ретривал ошибается, производит на работодателя
лучшее впечатление, чем работающее демо без единого числа рядом. Этот модуль
считает метрики по размеченному набору запросов (``eval/queries.yaml``) и
отдельно собирает провалы — запросы, на которых метод не нашёл ни одной
релевантной вакансии, вместе с тем, что он вернул вместо неё. Именно этот
список и идёт в README как раздел «где ретривал ошибается».
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from hh_radar.db.models import Vacancy
from hh_radar.rag.search import extract_vacancy_id, hybrid_search, semantic_search


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Одна размеченная строка запроса из ``eval/queries.yaml``."""

    id: str
    query: str
    intent: str
    relevant_vacancy_ids: tuple[int, ...]
    relevant_patterns: tuple[str, ...]


def load_queries(path: str | Path) -> list[QuerySpec]:
    """Читает и разбирает набор размеченных запросов из yaml."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        QuerySpec(
            id=str(item["id"]),
            query=str(item["query"]),
            intent=str(item.get("intent", "")),
            relevant_vacancy_ids=tuple(item.get("relevant_vacancy_ids") or []),
            relevant_patterns=tuple(item.get("relevant_patterns") or []),
        )
        for item in raw
    ]


def matches_relevant(vacancy: Vacancy, spec: QuerySpec) -> bool:
    """Считает вакансию релевантной запросу: приоритет — явная ручная разметка
    ``relevant_vacancy_ids``; если её нет (типичная ситуация, пока база не
    размечена вручную), запасной способ — совпадение названия вакансии хотя бы
    с одним regex из ``relevant_patterns``."""
    if vacancy.id in spec.relevant_vacancy_ids:
        return True
    return any(re.search(pattern, vacancy.name) for pattern in spec.relevant_patterns)


def _relevant_ids_for_spec(spec: QuerySpec, all_vacancies: Sequence[tuple[int, str]]) -> set[int]:
    """Строит множество id вакансий, релевантных запросу, по всей базе:
    ``relevant_vacancy_ids``, если разметка уже есть, иначе — регэкспы
    ``relevant_patterns`` против названий всех вакансий."""
    if spec.relevant_vacancy_ids:
        return set(spec.relevant_vacancy_ids)
    if not spec.relevant_patterns:
        return set()
    return {
        vacancy_id
        for vacancy_id, name in all_vacancies
        if any(re.search(pattern, name) for pattern in spec.relevant_patterns)
    }


def _reciprocal_rank(ranked_ids: list[int], relevant: set[int]) -> float:
    for position, vacancy_id in enumerate(ranked_ids, start=1):
        if vacancy_id in relevant:
            return 1.0 / position
    return 0.0


def _run_method(session: Session, method: str, query: str, top_k: int) -> tuple[list[int], float]:
    """Выполняет один метод поиска и замеряет время ответа.

    Полнотекстовый метод импортируется лениво, внутри функции: к моменту
    написания этого модуля ``hh_radar.db.queries.search_vacancies``
    (файл коллеги) мог ещё не существовать — тогда метод просто не находит
    ничего, вместо падения всей оценки.
    """
    started = time.perf_counter()
    if method == "fulltext":
        try:
            from hh_radar.db.queries import search_vacancies
        except ImportError:
            return [], time.perf_counter() - started
        results = search_vacancies(session, query, limit=top_k)
        ids = [extract_vacancy_id(item) for item in results]
    elif method == "semantic":
        ids = [hit.vacancy_id for hit in semantic_search(session, query, limit=top_k)]
    elif method == "hybrid":
        ids = [hit.vacancy_id for hit in hybrid_search(session, query, limit=top_k)]
    else:
        raise ValueError(f"Неизвестный метод поиска: {method!r}")
    return ids, time.perf_counter() - started


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    """Агрегированные по всему набору запросов метрики одного метода поиска."""

    method: str
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    avg_response_seconds: float
    queries_evaluated: int
    queries_skipped_no_ground_truth: int


@dataclass(frozen=True, slots=True)
class Failure:
    """Запрос, на котором метод не нашёл ни одной релевантной вакансии."""

    query_id: str
    query: str
    intent: str
    method: str
    expected_count: int
    returned: tuple[tuple[int, str], ...]  # (vacancy_id, name) того, что вернул метод


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Итог ``evaluate_retrieval``: метрики по методам плюс список провалов."""

    methods: tuple[MethodMetrics, ...]
    failures: tuple[Failure, ...]
    k_values: tuple[int, ...]
    queries_total: int


def evaluate_retrieval(
    session: Session,
    queries_path: str | Path,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    methods: tuple[str, ...] = ("fulltext", "semantic", "hybrid"),
) -> EvalReport:
    """Прогоняет размеченные запросы через каждый метод поиска и считает
    recall@k, precision@k, MRR и среднее время ответа.

    Запросы, для которых не удалось построить ни одного релевантного id
    (нет ни ручной разметки, ни совпадений по паттернам — например, база ещё
    не наполнена), пропускаются и не участвуют в усреднении метрик, но
    учитываются в ``queries_skipped_no_ground_truth``: молчаливо считать такой
    запрос провалом было бы нечестно — у него просто нет эталона для сравнения.

    Провалом считается запрос, на котором метод не нашёл ни одной релевантной
    вакансии среди top-5 (или среди top-k_max, если 5 нет в ``k_values``).
    Список провалов — это и есть материал для раздела README «где ретривал
    ошибается».
    """
    specs = load_queries(queries_path)
    max_k = max(k_values)
    fail_k = 5 if 5 in k_values else max_k

    # (id, name) всей базы — один раз на весь прогон, а не по разу на запрос:
    # именно так строится ground truth для запросов без ручной разметки.
    # .tuples() распаковывает Row в обычные tuple[int, str] — и rows не путаются
    # с типом Row при дальнейшей передаче в _relevant_ids_for_spec.
    all_vacancies = session.execute(select(Vacancy.id, Vacancy.name)).tuples().all()
    names_by_id: dict[int, str] = dict(all_vacancies)

    method_metrics: list[MethodMetrics] = []
    failures: list[Failure] = []

    for method in methods:
        recall_sums = dict.fromkeys(k_values, 0.0)
        precision_sums = dict.fromkeys(k_values, 0.0)
        mrr_sum = 0.0
        time_sum = 0.0
        evaluated = 0
        skipped = 0

        for spec in specs:
            relevant = _relevant_ids_for_spec(spec, all_vacancies)
            if not relevant:
                skipped += 1
                continue

            ranked_ids, elapsed = _run_method(session, method, spec.query, max_k)
            time_sum += elapsed
            evaluated += 1

            for k in k_values:
                top_k_hits = len(set(ranked_ids[:k]) & relevant)
                recall_sums[k] += top_k_hits / len(relevant)
                precision_sums[k] += top_k_hits / k

            mrr_sum += _reciprocal_rank(ranked_ids, relevant)

            if not (set(ranked_ids[:fail_k]) & relevant):
                failures.append(
                    Failure(
                        query_id=spec.id,
                        query=spec.query,
                        intent=spec.intent,
                        method=method,
                        expected_count=len(relevant),
                        returned=tuple(
                            (vid, names_by_id.get(vid, "?")) for vid in ranked_ids[:fail_k]
                        ),
                    )
                )

        if evaluated == 0:
            method_metrics.append(
                MethodMetrics(
                    method=method,
                    recall_at_k=dict.fromkeys(k_values, 0.0),
                    precision_at_k=dict.fromkeys(k_values, 0.0),
                    mrr=0.0,
                    avg_response_seconds=0.0,
                    queries_evaluated=0,
                    queries_skipped_no_ground_truth=skipped,
                )
            )
            continue

        method_metrics.append(
            MethodMetrics(
                method=method,
                recall_at_k={k: recall_sums[k] / evaluated for k in k_values},
                precision_at_k={k: precision_sums[k] / evaluated for k in k_values},
                mrr=mrr_sum / evaluated,
                avg_response_seconds=time_sum / evaluated,
                queries_evaluated=evaluated,
                queries_skipped_no_ground_truth=skipped,
            )
        )

    return EvalReport(
        methods=tuple(method_metrics),
        failures=tuple(failures),
        k_values=tuple(k_values),
        queries_total=len(specs),
    )


def render_report(report: EvalReport) -> str:
    """Markdown-отчёт: сравнительная таблица методов плюс раздел с провалами.
    Готов для вставки в README без правок."""
    lines: list[str] = [f"Запросов в размеченном наборе: {report.queries_total}.", ""]

    header = (
        ["Метод"]
        + [f"recall@{k}" for k in report.k_values]
        + [f"precision@{k}" for k in report.k_values]
        + ["MRR", "среднее время, с", "оценено запросов", "без ground truth"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in report.methods:
        row = (
            [m.method]
            + [f"{m.recall_at_k[k]:.2f}" for k in report.k_values]
            + [f"{m.precision_at_k[k]:.2f}" for k in report.k_values]
            + [
                f"{m.mrr:.2f}",
                f"{m.avg_response_seconds:.3f}",
                str(m.queries_evaluated),
                str(m.queries_skipped_no_ground_truth),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### Где ретривал ошибается")
    lines.append("")
    if not report.failures:
        lines.append("Провалов (0 релевантных в топ-5) на текущем наборе запросов не обнаружено.")
    else:
        for f in report.failures:
            lines.append(f"- **{f.query_id}** [{f.method}] «{f.query}» — {f.intent}")
            lines.append(f"  Ожидалось релевантных вакансий: {f.expected_count}. Вернул:")
            if f.returned:
                shown = ", ".join(f"{name} (id={vid})" for vid, name in f.returned[:5])
                lines.append(f"  {shown}")
            else:
                lines.append("  ничего.")

    return "\n".join(lines)


__all__: list[str] = [
    "EvalReport",
    "Failure",
    "MethodMetrics",
    "QuerySpec",
    "evaluate_retrieval",
    "load_queries",
    "matches_relevant",
    "render_report",
]
