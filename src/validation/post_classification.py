"""
Классификация оригинальных постов из evaluation_comparison.csv с помощью LLM.

Скрипт использует vLLM для батчевой категоризации реальных Telegram-постов
по тематике (tech_guide, case_study, product_update, market_news,
expert_opinion, other).

Предназначен для запуска в Google Colab с GPU.
"""

import pandas as pd
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ─── Настройки ────────────────────────────────────────────────────────────────

# Пути к данным (адаптируйте под Google Drive / Colab)
INPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison.csv'
OUTPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison_classified.csv'

# Модель-судья
JUDGE_MODEL = "google/gemma-4-26B-A4B-it"

# Допустимые категории для валидации
VALID_CATEGORIES = {
    "tech_guide", "case_study", "product_update",
    "market_news", "expert_opinion", "other",
}

# ─── Промт для классификации ──────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """Ты — Senior IT-маркетолог и дата-аналитик, специализирующийся на B2B-рынке информационных технологий.
Твоя задача — проанализировать текст Telegram-поста и отнести его к ОДНОЙ из строго заданных категорий.

КАТЕГОРИИ:
1. "tech_guide" (Технический гайд): Глубокий разбор технологий, архитектуры, пайплайнов, туториалы, best practices. Цель — обучение инженеров.
2. "case_study" (Кейс/Внедрение): Описание реального опыта решения бизнес-задачи клиента. Часто содержит метрики (ускорение процессов, снижение нагрузки, архитектурные решения).
3. "product_update" (Продукт/Анонс): Релизы новых фич, описание возможностей конкретного B2B-сервиса, интеграций, или приглашение на профильный вебинар/конференцию.
4. "market_news" (Новости/Аналитика): Обзор трендов рынка, аналитические отчеты, изменения в регулировании (например, импортозамещение).
5. "expert_opinion" (Мнение эксперта): Авторские размышления (CTO, Lead Engineer) об индустрии, холивары (например, микросервисы vs монолит) без прямой привязки к конкретному внедрению.
6. "other" (Другое): HR-контент (вакансии, жизнь команды), мемы или нерелевантный спам.

ПРАВИЛА:
- Выбери только ОДНУ наиболее подходящую категорию из списка выше.
- Твой ответ должен быть строго валидным JSON-объектом. Не добавляй никаких вступительных слов, markdown-разметки (кроме самого JSON) или пояснений вне объекта.

ФОРМАТ ОТВЕТА:
{
  "category": "название_категории_строго_из_списка_на_английском",
  "confidence": оценка_уверенности_от_1_до_10,
  "reasoning": "Краткое обоснование выбора (1 предложение)"
}"""


def format_prompts_for_judge(
    posts: list[str],
    tokenizer,
) -> list[str]:
    """Форматирует посты в формат чата модели для классификации."""
    formatted = []
    for post in posts:
        if not str(post).strip():
            formatted.append("")
            continue

        user_message = CLASSIFICATION_PROMPT + "\n\nТекст поста:\n" + str(post)

        messages = [
            {"role": "user", "content": user_message},
        ]

        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(chat_prompt)

    return formatted


def parse_json_response(response: str) -> dict:
    """
    Извлекает JSON-классификацию из ответа LLM.
    Поддерживает случаи, когда модель добавляет текст вокруг JSON.
    """
    default = {
        "category": None,
        "confidence": None,
        "reasoning": None,
    }

    if not response or not response.strip():
        return default

    # Пытаемся найти JSON в ответе
    json_match = re.search(r'\{[^{}]*\}', response)
    if not json_match:
        return default

    try:
        parsed = json.loads(json_match.group())

        category = parsed.get("category", "").strip().lower()
        if category not in VALID_CATEGORIES:
            category = None

        confidence = parsed.get("confidence")
        if confidence is not None:
            confidence = int(confidence)
            confidence = max(1, min(10, confidence))

        reasoning = parsed.get("reasoning")
        if reasoning is not None:
            reasoning = str(reasoning).strip()

        return {
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def main():
    # ── 1. Загрузка данных ────────────────────────────────────────────────────
    print(f"Загрузка данных из {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, sep=';', encoding='utf-8')
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {df.columns.tolist()}")

    real_posts = df['real_post'].fillna("").tolist()

    # ── 2. Инициализация токенизатора ─────────────────────────────────────────
    print(f"\nИнициализация токенизатора для {JUDGE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

    # ── 3. Форматирование промтов ─────────────────────────────────────────────
    print("Подготовка промтов для классификации...")
    formatted_prompts = format_prompts_for_judge(real_posts, tokenizer)

    # Отфильтровываем пустые (запоминаем индексы непустых)
    valid_indices = [i for i, p in enumerate(formatted_prompts) if p.strip()]
    valid_prompts = [formatted_prompts[i] for i in valid_indices]
    print(f"Непустых записей для классификации: {len(valid_prompts)} из {len(formatted_prompts)}")

    # ── 4. Инициализация vLLM ─────────────────────────────────────────────────
    print(f"\nИнициализация vLLM с моделью {JUDGE_MODEL}...")
    llm = LLM(
        model=JUDGE_MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        #quantization='bitsandbytes',     # 4-bit квантизация (NF4)
        #load_format='bitsandbytes',
    )

    # Настройки генерации (низкая температура для стабильной классификации)
    sampling_params = SamplingParams(
        temperature=1,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.0,
    )

    # ── 5. Генерация классификаций ────────────────────────────────────────────
    print(f"\nНачало классификации {len(valid_prompts)} постов...")
    outputs = llm.generate(valid_prompts, sampling_params)

    # ── 6. Парсинг результатов ────────────────────────────────────────────────
    print("Парсинг ответов модели...")

    # Инициализируем колонки
    df['category'] = None
    df['confidence'] = None
    df['reasoning'] = None
    df['raw_response'] = ""

    parsed_count = 0
    failed_count = 0

    for idx, output in zip(valid_indices, outputs):
        response_text = output.outputs[0].text.strip()
        df.at[idx, 'raw_response'] = response_text

        result = parse_json_response(response_text)

        if result['category'] is not None:
            parsed_count += 1
        else:
            failed_count += 1

        df.at[idx, 'category'] = result['category']
        df.at[idx, 'confidence'] = result['confidence']
        df.at[idx, 'reasoning'] = result['reasoning']

    print(f"\nУспешно распарсено: {parsed_count}")
    print(f"Не удалось распарсить: {failed_count}")

    # ── 7. Статистика ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ ОРИГИНАЛЬНЫХ ПОСТОВ")
    print("=" * 60)

    category_counts = df['category'].value_counts()
    total_classified = category_counts.sum()

    for cat in VALID_CATEGORIES:
        count = category_counts.get(cat, 0)
        pct = count / total_classified * 100 if total_classified > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n  {'ИТОГО':20s}: {total_classified}")

    # Средняя уверенность по категориям
    print("\nСредняя уверенность (confidence) по категориям:")
    confidence_by_cat = df.groupby('category')['confidence'].mean()
    for cat, conf in confidence_by_cat.items():
        print(f"  {cat:20s}: {conf:.1f}/10")

    print("=" * 60)

    # ── 8. Сохранение ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, sep=';', index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в {OUTPUT_PATH}")
    print(f"Добавлены колонки: ['category', 'confidence', 'reasoning', 'raw_response']")


if __name__ == "__main__":
    main()
