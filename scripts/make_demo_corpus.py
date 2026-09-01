"""Генератор демонстрационного корпуса вакансий.

Зачем он нужен. Вакансии hh нельзя ни выложить в публичный репозиторий, ни
скачать без токена приложения: ``GET /vacancies`` отвечает анонимному клиенту
``403``. Значит, у человека, который просто открыл ссылку на витрину или
склонировал репозиторий, данных нет вообще — а на четырёх примерах любой график
показывает «1 шт.» и не даёт понять, работает система или нет.

Поэтому корпус собирается программой. Вакансии **вымышлены целиком**:
работодатели, тексты, зарплаты, идентификаторы. Это не выгрузка с hh и не
попытка её изобразить — числа, посчитанные по этому корпусу, ничего не говорят
о настоящем рынке, и витрина честно предупреждает об этом баннером.

Ценность в другом: корпус проходит ровно через тот же ``parse_vacancy``, тот же
конвейер записи, тот же чанкинг и те же эмбеддинги, что и живые данные. То есть
он показывает работу системы, не подменяя собой её предмет.

Идентификаторы взяты из диапазона ``900_000_000+``: на hh сейчас девятизначные
номера порядка ``136_000_000``, так что демонстрационная вакансия не может
совпасть с настоящей.

Генерация детерминированная: при одном и том же ``--seed`` файл получается
побайтово одинаковым, и в истории git не шумят случайные диффы.

    python scripts/make_demo_corpus.py --out samples/03-demo-corpus.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

MSK = timezone(timedelta(hours=3))

#: Первый идентификатор демонстрационной вакансии. Заведомо выше настоящих.
FIRST_ID = 900_000_001

#: Столько вакансий в корпусе по умолчанию. Число выбрано между двух границ:
#: графики уже осмысленные, но до порога ``MIN_MEANINGFUL_VACANCIES`` витрины
#: далеко — баннер «это демонстрационные данные» продолжает показываться.
DEFAULT_COUNT = 48


# ------------------------------------------------------------- справочники --


AREAS: list[dict[str, str]] = [
    {"id": "1", "name": "Москва"},
    {"id": "2", "name": "Санкт-Петербург"},
    {"id": "3", "name": "Екатеринбург"},
    {"id": "4", "name": "Новосибирск"},
    {"id": "88", "name": "Казань"},
    {"id": "66", "name": "Нижний Новгород"},
    {"id": "76", "name": "Ростов-на-Дону"},
    {"id": "104", "name": "Пермь"},
]

#: Вымышленные работодатели. Совпадение названия с существующей компанией было
#: бы случайным: ни данных, ни отношения к ней этот корпус не имеет.
EMPLOYERS: list[dict[str, Any]] = [
    {"id": 8100001, "name": "Кедр Диджитал", "trusted": True, "accredited": True},
    {"id": 8100002, "name": "Полярис Ритейл", "trusted": True, "accredited": False},
    {"id": 8100003, "name": "Сигма Практика", "trusted": True, "accredited": True},
    {"id": 8100004, "name": "Тайга Лабс", "trusted": False, "accredited": False},
    {"id": 8100005, "name": "Мосты Финтех", "trusted": True, "accredited": True},
    {"id": 8100006, "name": "Гринфилд Агро", "trusted": True, "accredited": False},
    {"id": 8100007, "name": "Оникс Телеком", "trusted": True, "accredited": True},
    {"id": 8100008, "name": "Дельта Букинг", "trusted": False, "accredited": False},
    {"id": 8100009, "name": "Лига Образования", "trusted": True, "accredited": False},
    {"id": 8100010, "name": "Речной Порт Групп", "trusted": True, "accredited": False},
    {"id": 8100011, "name": "Астра Клиники", "trusted": True, "accredited": True},
    {"id": 8100012, "name": "Юнион Мебель", "trusted": False, "accredited": False},
    {"id": 8100013, "name": "Прайм Логистика", "trusted": True, "accredited": True},
    {"id": 8100014, "name": "Сфера Медиа", "trusted": True, "accredited": False},
]

EXPERIENCE = {
    "noExperience": "Нет опыта",
    "between1And3": "От 1 года до 3 лет",
    "between3And6": "От 3 до 6 лет",
    "moreThan6": "Более 6 лет",
}

CONDITIONS = [
    "Работаем удалённо, график гибкий, созвон один раз в день.",
    "Гибрид: два дня в офисе, остальное из дома.",
    "Офис в центре, но на удалёнку отпускаем без вопросов.",
    "Оформление по ТК, ДМС после испытательного срока.",
    "Небольшая команда, решения принимаются за день, а не за квартал.",
    "Есть бюджет на обучение и конференции.",
    "Испытательный срок — два месяца, зарплата на нём не режется.",
]


# ------------------------------------------------------------- архетипы ролей --
#
# Каждый архетип — это набор кирпичей, из которых собирается вакансия. Тексты
# намеренно разные по формулировкам: если бы все объявления про одно и то же
# были написаны одними словами, семантический поиск на таком корпусе выглядел
# бы лучше, чем он есть, — а это ровно тот самообман, ради избавления от
# которого в проекте вообще заведена оценка качества.

ROLES: list[dict[str, Any]] = [
    {
        "key": "n8n",
        "titles": [
            "Инженер по автоматизации (n8n)",
            "Разработчик интеграций n8n",
            "Automation Engineer (n8n / Make)",
            "Специалист по автоматизации бизнес-процессов",
        ],
        "skills": ["n8n", "Python", "REST API", "Docker", "PostgreSQL", "Webhook", "JavaScript"],
        "salary": (110_000, 230_000),
        "experience": ["noExperience", "between1And3", "between1And3", "between3And6"],
        "duties": [
            "Собирать сценарии в n8n: от выгрузки заявок до уведомлений в мессенджеры",
            "Писать кастомные ноды на TypeScript, когда готовых не хватает",
            "Подключать внутренние сервисы по REST API и вебхукам",
            "Следить за очередями и разбирать упавшие запуски",
            "Переносить существующие сценарии с Make на self-hosted n8n",
        ],
        "reqs": [
            "Опыт с n8n, Make или Zapier от полугода",
            "Python или JavaScript на уровне уверенного скриптования",
            "Понимание REST, вебхуков и авторизации по токену",
            "Docker на уровне «поднять, посмотреть логи, починить»",
        ],
    },
    {
        "key": "llm",
        "titles": [
            "AI-инженер (LLM)",
            "Инженер LLM-решений",
            "Разработчик AI-агентов",
            "LLM Engineer",
        ],
        "skills": ["Python", "LLM", "RAG", "LangChain", "OpenAI API", "PostgreSQL", "Docker"],
        "salary": (150_000, 320_000),
        "experience": ["between1And3", "between1And3", "between3And6", "between3And6"],
        "duties": [
            "Встраивать языковые модели во внутренние процессы компании",
            "Собирать RAG-пайплайны поверх корпоративной базы знаний",
            "Проектировать и отлаживать промпты, мерить качество ответов",
            "Подключать модели к внешним инструментам через функции и MCP",
            "Считать стоимость запросов и держать её в рамках бюджета",
        ],
        "reqs": [
            "Python на уровне продакшен-кода, а не блокнотов",
            "Опыт с API языковых моделей и понимание их ограничений",
            "Знакомство с векторным поиском: pgvector, Qdrant или аналоги",
            "Умение объяснить, почему модель ответила именно так",
        ],
    },
    {
        "key": "support",
        "titles": [
            "Инженер поддержки автоматизаций",
            "Разработчик доработок интеграций",
            "Специалист по сопровождению интеграций",
        ],
        "skills": ["Python", "REST API", "SQL", "Git", "Docker", "Linux"],
        "salary": (90_000, 180_000),
        "experience": ["noExperience", "between1And3", "between1And3"],
        "duties": [
            "Разбираться в чужом коде интеграций, который писали до вас",
            "Чинить сценарии, сломавшиеся после обновления внешнего API",
            "Дорабатывать существующие процессы под новые требования отделов",
            "Восстанавливать работу выгрузок после сбоев и разбирать причины",
            "Приводить в порядок то, что собрано наспех и держится на костылях",
        ],
        "reqs": [
            "Готовность работать с легаси, а не переписывать всё с нуля",
            "Python или PHP на уровне чтения и правки чужого кода",
            "Умение читать логи и воспроизводить ошибку по описанию пользователя",
            "Спокойное отношение к задачам вида «вчера работало, сегодня нет»",
        ],
    },
    {
        "key": "bot",
        "titles": [
            "Разработчик чат-ботов",
            "Python-разработчик (Telegram-боты)",
            "Инженер диалоговых сервисов",
        ],
        "skills": ["Python", "aiogram", "Telegram Bot API", "PostgreSQL", "Docker", "Redis"],
        "salary": (100_000, 210_000),
        "experience": ["noExperience", "between1And3", "between1And3", "between3And6"],
        "duties": [
            "Разрабатывать ботов для поддержки клиентов и внутренних заявок",
            "Подключать к ботам языковые модели и базу знаний компании",
            "Держать состояние диалога и переводить сложные случаи на оператора",
            "Собирать метрики: сколько вопросов бот закрыл сам",
        ],
        "reqs": [
            "Python и asyncio, опыт с aiogram или telebot",
            "PostgreSQL или Redis для хранения состояния",
            "Понимание, чем бот с сценарием отличается от бота с моделью",
        ],
    },
    {
        "key": "data",
        "titles": [
            "Инженер данных",
            "Data Engineer",
            "Разработчик ETL-процессов",
        ],
        "skills": ["Python", "SQL", "Airflow", "PostgreSQL", "ClickHouse", "Docker", "ETL"],
        "salary": (160_000, 340_000),
        "experience": ["between1And3", "between3And6", "between3And6", "moreThan6"],
        "duties": [
            "Строить и поддерживать пайплайны загрузки данных в хранилище",
            "Описывать расчёты витрин и следить за их корректностью",
            "Оптимизировать тяжёлые запросы и планы выполнения",
            "Настраивать мониторинг свежести данных",
        ],
        "reqs": [
            "SQL на уровне оконных функций и разбора планов",
            "Python для оркестрации, Airflow или аналог",
            "Опыт с колоночными хранилищами будет плюсом",
        ],
    },
    {
        "key": "mlops",
        "titles": [
            "MLOps-инженер",
            "Инженер ML-инфраструктуры",
            "MLOps / Platform Engineer",
        ],
        "skills": ["Docker", "Kubernetes", "Python", "CI/CD", "MLflow", "Linux", "Terraform"],
        "salary": (200_000, 400_000),
        "experience": ["between3And6", "between3And6", "moreThan6"],
        "duties": [
            "Заворачивать модели в сервисы и выкатывать их в кластер",
            "Держать воспроизводимость экспериментов и версионирование весов",
            "Настраивать сборку, тесты и выкладку через CI",
            "Отвечать за доступность инференса под нагрузкой",
        ],
        "reqs": [
            "Kubernetes на уровне эксплуатации, а не курсов",
            "Python и понимание жизненного цикла модели",
            "Опыт с очередями и мониторингом",
        ],
    },
    {
        "key": "crm",
        "titles": [
            "Интегратор CRM (amoCRM / Битрикс24)",
            "Разработчик интеграций CRM",
            "Специалист по внедрению CRM",
        ],
        "skills": ["amoCRM", "Битрикс24", "REST API", "PHP", "Python", "Webhook"],
        "salary": (80_000, 190_000),
        "experience": ["noExperience", "between1And3", "between1And3"],
        "duties": [
            "Настраивать воронки, поля и права в CRM под процессы отдела продаж",
            "Связывать CRM с телефонией, почтой и складским учётом",
            "Писать обработчики вебхуков для нестандартных сценариев",
            "Переносить данные из старых систем без потерь",
        ],
        "reqs": [
            "Опыт внедрения amoCRM или Битрикс24",
            "REST API и вебхуки на практике",
            "PHP или Python для обработчиков",
        ],
    },
    {
        "key": "backend",
        "titles": [
            "Python-разработчик (бэкенд автоматизаций)",
            "Backend-разработчик Python",
            "Разработчик внутренних сервисов",
        ],
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST API", "Git", "SQLAlchemy"],
        "salary": (140_000, 300_000),
        "experience": ["between1And3", "between3And6", "between3And6"],
        "duties": [
            "Разрабатывать сервисы, на которые опираются внутренние автоматизации",
            "Проектировать схему базы и писать миграции",
            "Покрывать код тестами и держать их зелёными",
            "Разбирать инциденты на проде вместе с командой",
        ],
        "reqs": [
            "Python и FastAPI или Django",
            "PostgreSQL: индексы, транзакции, планы запросов",
            "Docker и понимание, как код доезжает до сервера",
        ],
    },
    {
        "key": "rpa",
        "titles": [
            "RPA-разработчик",
            "Инженер роботизации процессов",
            "RPA Developer",
        ],
        "skills": ["RPA", "Python", "Selenium", "SQL", "1С"],
        "salary": (110_000, 240_000),
        "experience": ["between1And3", "between1And3", "between3And6"],
        "duties": [
            "Роботизировать рутину бухгалтерии и кадрового учёта",
            "Разбирать входящие документы и переносить данные между системами",
            "Поддерживать роботов, когда интерфейсы систем меняются",
        ],
        "reqs": [
            "Опыт с RPA-платформами или собственными скриптами",
            "Python, работа с таблицами и документами",
            "Внимание к краевым случаям: робот не догадается, человек догадается",
        ],
    },
    {
        "key": "analyst",
        "titles": [
            "Аналитик данных (с AI-инструментами)",
            "Продуктовый аналитик",
            "Аналитик автоматизации",
        ],
        "skills": ["SQL", "Python", "BI", "Excel", "Pandas"],
        "salary": (90_000, 200_000),
        "experience": ["noExperience", "between1And3", "between1And3", "between3And6"],
        "duties": [
            "Считать метрики процессов и показывать, где теряется время",
            "Собирать дашборды, на которые смотрит руководитель, а не аналитик",
            "Оценивать эффект автоматизации до и после внедрения",
        ],
        "reqs": [
            "SQL уверенно, Python на уровне pandas",
            "Умение задать вопрос данным раньше, чем строить график",
        ],
    },
    {
        "key": "ml",
        "titles": [
            "ML-инженер (NLP)",
            "Инженер машинного обучения",
            "ML Engineer",
        ],
        "skills": ["Python", "PyTorch", "NLP", "Transformers", "Docker", "SQL"],
        "salary": (180_000, 380_000),
        "experience": ["between1And3", "between3And6", "moreThan6"],
        "duties": [
            "Обучать и дообучать модели под задачи классификации обращений",
            "Готовить и чистить обучающие выборки",
            "Мерить качество на отложенной выборке и не обманывать себя метрикой",
        ],
        "reqs": [
            "Python и PyTorch",
            "Понимание, чем валидация отличается от теста",
            "Опыт с трансформерами для русского языка",
        ],
    },
    {
        "key": "prompt",
        "titles": [
            "Промпт-инженер",
            "Специалист по работе с LLM",
            "AI-редактор процессов",
        ],
        "skills": ["LLM", "Промпт-инжиниринг", "Python", "Анализ данных"],
        "salary": (80_000, 190_000),
        "experience": ["noExperience", "noExperience", "between1And3"],
        "duties": [
            "Писать и отлаживать промпты для внутренних сценариев",
            "Собирать наборы примеров и проверять на них ответы модели",
            "Обучать коллег пользоваться готовыми сценариями",
        ],
        "reqs": [
            "Опыт работы с языковыми моделями в рабочих задачах",
            "Аккуратность в формулировках и умение мерить результат",
            "Базовый Python — плюс, но не обязателен",
        ],
    },
]

INTROS = [
    "Компания «{employer}» ищет специалиста в команду внутренней автоматизации.",
    "Мы в «{employer}» строим внутренние сервисы и ищем человека, который их разовьёт.",
    "«{employer}» расширяет команду разработки: нужен человек на задачи ниже.",
    "В «{employer}» набирается объём ручной работы, который пора автоматизировать.",
]


# ------------------------------------------------------------------ сборка --


def build_vacancy(rng: random.Random, index: int, published: datetime) -> dict[str, Any]:
    """Собрать одну вакансию в том формате, в котором её отдаёт карточка hh."""
    role = ROLES[index % len(ROLES)]
    employer = rng.choice(EMPLOYERS)
    area = rng.choice(AREAS)
    vacancy_id = FIRST_ID + index

    duties = rng.sample(role["duties"], k=min(len(role["duties"]), rng.randint(3, 4)))
    reqs = rng.sample(role["reqs"], k=min(len(role["reqs"]), rng.randint(2, 3)))
    skills = rng.sample(role["skills"], k=min(len(role["skills"]), rng.randint(3, 5)))

    description = (
        f"<p>{rng.choice(INTROS).format(employer=employer['name'])}</p> "
        "<p><strong>Чем предстоит заниматься:</strong></p> <ul>"
        + "".join(f"<li>{item}</li>" for item in duties)
        + "</ul> <p><strong>Ждём от вас:</strong></p> <ul>"
        + "".join(f"<li>{item}</li>" for item in reqs)
        + f"</ul><br/><p>{rng.choice(CONDITIONS)}</p>"
    )

    experience_id = rng.choice(role["experience"])
    remote = rng.random() < 0.55

    # hh отдаёт вилку сразу в двух полях: старом `salary` и новом
    # `salary_range` с режимом выплаты. Разбор предпочитает новое, но читает
    # оба — корпус повторяет эту двойственность, иначе запасная ветка кода
    # никогда не встретилась бы с данными.
    salary = _salary(rng, role)
    salary_range = None
    if salary:
        salary_range = dict(salary, mode={"id": "MONTH", "name": "За месяц"}, frequency=None)

    return {
        "id": str(vacancy_id),
        "premium": False,
        "name": rng.choice(role["titles"]),
        "area": {
            "id": area["id"],
            "name": area["name"],
            "url": f"https://api.hh.ru/areas/{area['id']}",
        },
        "salary": salary,
        "salary_range": salary_range,
        "type": {"id": "open", "name": "Открытая"},
        "address": None,
        "published_at": published.isoformat(timespec="seconds").replace("+03:00", "+0300"),
        "created_at": published.isoformat(timespec="seconds").replace("+03:00", "+0300"),
        "archived": False,
        "url": f"https://api.hh.ru/vacancies/{vacancy_id}",
        "alternate_url": f"https://hh.ru/vacancy/{vacancy_id}",
        "apply_alternate_url": f"https://hh.ru/applicant/vacancy_response?vacancyId={vacancy_id}",
        "relations": [],
        "employer": {
            "id": str(employer["id"]),
            "name": employer["name"],
            "url": f"https://api.hh.ru/employers/{employer['id']}",
            "alternate_url": f"https://hh.ru/employer/{employer['id']}",
            "vacancies_url": f"https://api.hh.ru/vacancies?employer_id={employer['id']}",
            "accredited_it_employer": employer["accredited"],
            "trusted": employer["trusted"],
        },
        "contacts": None,
        "schedule": {"id": "remote", "name": "Удаленная работа"}
        if remote
        else {"id": "fullDay", "name": "Полный день"},
        "work_format": [{"id": "REMOTE", "name": "Удалённо"}]
        if remote
        else [{"id": "ON_SITE", "name": "На месте работодателя"}],
        "professional_roles": [{"id": "96", "name": "Программист, разработчик"}],
        "experience": {"id": experience_id, "name": EXPERIENCE[experience_id]},
        "employment": {"id": "full", "name": "Полная занятость"},
        "description": description,
        "key_skills": [{"name": name} for name in skills],
        "accept_handicapped": False,
        "accept_kids": False,
        "languages": [],
    }


def _salary(rng: random.Random, role: dict[str, Any]) -> dict[str, Any] | None:
    """Вилка. У каждой пятой вакансии её нет — как и на настоящем hh."""
    if rng.random() < 0.2:
        return None

    low, high = role["salary"]
    start = rng.randrange(low, high, 10_000)
    has_upper = rng.random() < 0.65
    upper = start + rng.randrange(20_000, 90_000, 10_000) if has_upper else None

    #: Две вакансии из корпуса намеренно в валюте: приведение к рублям — часть
    #: разбора, и оно должно быть видно в демонстрации, а не только в тестах.
    currency = rng.choices(["RUR", "USD", "KZT"], weights=[92, 5, 3], k=1)[0]
    if currency == "USD":
        start, upper = start // 90, (upper // 90 if upper else None)
    elif currency == "KZT":
        start, upper = int(start / 0.19), (int(upper / 0.19) if upper else None)

    return {
        "from": start,
        "to": upper,
        "currency": currency,
        "gross": rng.random() < 0.6,
    }


def generate(count: int, until: date, days: int, seed: int) -> list[dict[str, Any]]:
    """Собрать корпус целиком.

    Даты публикации распределяются по интервалу неравномерно: в будни вакансий
    больше, чем в выходные. График «сколько публикуют по неделям» на витрине
    из-за этого выглядит как настоящий, а не как ровная полка.
    """
    rng = random.Random(seed)
    start = until - timedelta(days=days)
    vacancies: list[dict[str, Any]] = []

    for index in range(count):
        offset = rng.randint(0, days)
        day = start + timedelta(days=offset)
        if day.weekday() >= 5 and rng.random() < 0.7:
            day -= timedelta(days=rng.randint(1, 2))
        moment = datetime.combine(
            day, time(hour=rng.randint(9, 19), minute=rng.choice([0, 12, 15, 30, 45])), tzinfo=MSK
        )
        vacancies.append(build_vacancy(rng, index, moment))

    vacancies.sort(key=lambda item: item["published_at"], reverse=True)
    return vacancies


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("samples/03-demo-corpus.json"))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--days", type=int, default=30, help="глубина интервала публикации")
    parser.add_argument(
        "--until",
        type=date.fromisoformat,
        default=date.today(),
        help="последний день интервала, ГГГГ-ММ-ДД",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    vacancies = generate(args.count, args.until, args.days, args.seed)
    payload = {
        "_comment": (
            "Демонстрационный корпус. Вакансии, работодатели и зарплаты вымышлены "
            "и сгенерированы scripts/make_demo_corpus.py. Это не выгрузка с hh.ru: "
            "по этим числам нельзя судить о рынке."
        ),
        "items": vacancies,
        "found": len(vacancies),
        "pages": 1,
        "page": 0,
        "per_page": len(vacancies),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"{args.out}: {len(vacancies)} вакансий")


if __name__ == "__main__":
    main()
