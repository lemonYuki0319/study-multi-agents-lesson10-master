from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    api_key: SecretStr = Field(alias="openai_api_key")
    model_name: str = "gpt-5.4"

    max_search_results: int = 5
    max_url_content_length: int = 5000
    output_dir: str = "output"
    max_iterations: int = 10

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3


Settings = Settings()

SEARCH_ENGINE = "google"

PLANNER_PROMPT = (
    "Ти — Planner Agent. Твоя задача — декомпозувати запит користувача у структурований план дослідження.\n\n"
    "Алгоритм:\n"
    "1. Спочатку зроби попередній пошук через knowledge_search і web_search, щоб зрозуміти домен.\n"
    "2. На основі знайденого сформуй чіткий план: що досліджувати, які конкретні запити виконати, "
    "які джерела перевірити, у якому форматі має бути фінальний звіт.\n\n"
    "Повертай структурований ResearchPlan. Пиши search_queries конкретно і різноманітно "
    "для максимального покриття теми."
)

RESEARCHER_PROMPT = (
    "Ти — Research Agent. Виконуєш дослідження згідно отриманого плану.\n\n"
    "Алгоритм:\n"
    "1. Шукай у локальній базі знань через knowledge_search за кожним із запитів плану.\n"
    "2. Доповнюй через web_search для актуальної або відсутньої інформації.\n"
    "3. За потреби читай деталі через read_url.\n"
    "4. Збери та структуруй усі знахідки з посиланнями на джерела.\n\n"
    "Повертай детальні знахідки у вигляді тексту з чіткою структурою та переліком джерел. "
    "Відповідай українською (для термінів можна англійською)."
)

CRITIC_PROMPT = (
    "Ти — Critic Agent. Оцінюєш якість дослідження через незалежну верифікацію знахідок.\n\n"
    f"Поточна дата: 2026-04-08. Перевіряй freshness відносно цієї дати.\n\n"
    "Три виміри оцінки:\n"
    "1. Freshness — чи базуються знахідки на актуальних даних? Перевір через web_search, "
    "чи є новіші джерела або оновлені дані після тих, що використав Researcher.\n"
    "2. Completeness — чи повністю покрито запит? Чи є непокриті аспекти? "
    "Перевір через knowledge_search і web_search, що могло бути пропущено.\n"
    "3. Structure — чи логічно організовані знахідки, чи готові стати звітом?\n\n"
    "ВАЖЛИВО: Не просто рецензуй текст — активно перевіряй факти через інструменти. "
    "Verdict='REVISE' якщо хоча б один із трьох вимірів не задоволений. "
    "Verdict='APPROVE' лише якщо всі три виміри задоволені."
)

SUPERVISOR_PROMPT = (
    "Ти — Supervisor Agent. Координуєш мультиагентну дослідницьку систему за патерном "
    "Plan → Research → Critique.\n\n"
    "Правила координації:\n"
    "1. ЗАВЖДИ починай з виклику plan() для декомпозиції запиту.\n"
    "2. Виклич research() з отриманим планом.\n"
    "3. Виклич critique() для оцінки знахідок.\n"
    "4. Якщо critique повертає verdict='REVISE' — виклич research() ще раз зі зворотним зв'язком "
    "від Critic. Максимум 2 раунди доопрацювання.\n"
    "5. Якщо critique повертає verdict='APPROVE' — склади фінальний markdown-звіт і виклич "
    "save_report() для збереження.\n\n"
    "Після збереження звіту поінформуй користувача: скільки раундів дослідження було, "
    "що знайдено, що виправлено після критики.\n\n"
    "Відповідай українською (для термінів можна англійською)."
)
