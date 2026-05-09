"""
Ранжирование сгенерированных постов с помощью LLM (слепая оценка).

Для каждого промта берёт три генерации (base, sft, ipo), случайно перемешивает
их (присваивая временные ID «Текст 1/2/3»), и просит модель-судью отранжировать
их от лучшего к худшему. Категория поста берётся из результатов классификации.

Предназначен для запуска в Google Colab с GPU.
"""

import pandas as pd
import json
import re
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ─── Настройки ────────────────────────────────────────────────────────────────

# Входной файл — результат post_classification.py (содержит колонку category)
INPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison_classified.csv'
OUTPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison_ranked.csv'

# Модель-судья (отличается от моделей-генераторов для объективности)
JUDGE_MODEL = "google/gemma-4-26B-A4B-it"

# Для воспроизводимости перемешивания
RANDOM_SEED = 42

# Маппинг категорий на человекочитаемые названия
CATEGORY_NAMES = {
    "tech_guide": "Технический гайд",
    "case_study": "Кейс/Внедрение",
    "product_update": "Продукт/Анонс",
    "market_news": "Новости/Аналитика",
    "expert_opinion": "Мнение эксперта",
    "other": "Другое",
}

# Колонки с генерациями и их внутренние ключи
MODEL_COLUMNS = {
    "generated_base": "base",
    "generated_sft": "sft",
    "generated_ipo": "ipo",
}

# ─── Промт для ранжирования ──────────────────────────────────────────────────

RANKING_PROMPT_TEMPLATE = """Ты — Senior IT-маркетолог. Оцени три варианта Telegram-поста для B2B ИТ-аудитории.
Категория поста: {category_name}

Критерии оценки:
1. Отсутствие 'инфоцыганства' и пустых восторгов.
2. Точность терминологии (K8s, CI/CD, SLA и т.д.).
3. Наличие четкой ценности для бизнеса (ROI, оптимизация, безопасность).
4. Читаемость и структура (уместность эмодзи, абзацы).

[Текст 1]
{text_1}

[Текст 2]
{text_2}

[Текст 3]
{text_3}

Твоя задача — отранжировать тексты от лучшего (1 место) к худшему (3 место). 
Ответ выдай строго в формате JSON:
{{
  "1st_place": "Текст X",
  "2nd_place": "Текст Y",
  "3rd_place": "Текст Z",
  "reasoning": "Краткое обоснование, почему победитель лучше остальных, и в чем главная ошибка текста на 3-м месте."
}}"""


def shuffle_texts(
    base_text: str,
    sft_text: str,
    ipo_text: str,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Перемешивает три текста и возвращает:
    - shuffled: список из (label, text), например [("Текст 1", "..."), ...]
    - label_to_model: маппинг label -> model_key, например {"Текст 1": "sft", ...}
    """
    items = [
        ("base", str(base_text)),
        ("sft", str(sft_text)),
        ("ipo", str(ipo_text)),
    ]
    rng.shuffle(items)

    shuffled = []
    label_to_model = {}
    for i, (model_key, text) in enumerate(items, start=1):
        label = f"Текст {i}"
        shuffled.append((label, text))
        label_to_model[label] = model_key

    return shuffled, label_to_model


def build_ranking_prompt(category: str, shuffled_texts: list[tuple[str, str]]) -> str:
    """Собирает финальный промт для ранжирования."""
    category_name = CATEGORY_NAMES.get(category, category or "Неизвестная")

    return RANKING_PROMPT_TEMPLATE.format(
        category_name=category_name,
        text_1=shuffled_texts[0][1],
        text_2=shuffled_texts[1][1],
        text_3=shuffled_texts[2][1],
    )


def format_prompts_for_judge(
    df: pd.DataFrame,
    tokenizer,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Форматирует все строки DataFrame в промты для модели-судьи.
    
    Возвращает:
    - formatted_prompts: список отформатированных промтов
    - all_mappings: список маппингов label->model для каждой строки
    """
    rng = random.Random(RANDOM_SEED)
    formatted_prompts = []
    all_mappings = []

    for _, row in df.iterrows():
        base_text = row.get('generated_base', '')
        sft_text = row.get('generated_sft', '')
        ipo_text = row.get('generated_ipo', '')
        category = row.get('category', 'other')

        # Пропускаем строки, где хотя бы одна генерация пустая
        if (not str(base_text).strip()
                or not str(sft_text).strip()
                or not str(ipo_text).strip()):
            formatted_prompts.append("")
            all_mappings.append({})
            continue

        shuffled, label_to_model = shuffle_texts(base_text, sft_text, ipo_text, rng)
        user_message = build_ranking_prompt(str(category), shuffled)

        messages = [
            {"role": "user", "content": user_message},
        ]

        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(chat_prompt)
        all_mappings.append(label_to_model)

    return formatted_prompts, all_mappings


def parse_ranking_response(
    response: str,
    label_to_model: dict[str, str],
) -> dict:
    """
    Извлекает ранжирование из JSON-ответа модели и маппит обратно на модели.

    Возвращает dict с ключами:
      rank_1, rank_2, rank_3 — имена моделей (base/sft/ipo)
      reasoning — обоснование
      raw_response — исходный ответ
      parse_success — удалось ли распарсить
    """
    default = {
        "rank_1": None,
        "rank_2": None,
        "rank_3": None,
        "reasoning": None,
        "raw_response": response,
        "parse_success": False,
    }

    if not response or not response.strip() or not label_to_model:
        return default

    # Ищем JSON в ответе
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if not json_match:
        return default

    try:
        parsed = json.loads(json_match.group())

        # Извлекаем места
        places = {}
        for place_key, rank_key in [("1st_place", "rank_1"),
                                     ("2nd_place", "rank_2"),
                                     ("3rd_place", "rank_3")]:
            label = parsed.get(place_key, "").strip()
            # Нормализация: "Текст 1", "текст 1", "Text 1" и т.д.
            text_num = re.search(r'(\d)', label)
            if text_num:
                normalized_label = f"Текст {text_num.group(1)}"
                model = label_to_model.get(normalized_label)
                places[rank_key] = model
            else:
                places[rank_key] = None

        reasoning = parsed.get("reasoning", "")

        # Проверяем, что все 3 места заполнены разными моделями
        models = [places.get("rank_1"), places.get("rank_2"), places.get("rank_3")]
        if None in models or len(set(models)) != 3:
            return default

        return {
            "rank_1": places["rank_1"],
            "rank_2": places["rank_2"],
            "rank_3": places["rank_3"],
            "reasoning": str(reasoning).strip(),
            "raw_response": response,
            "parse_success": True,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def main():
    # ── 1. Загрузка данных ────────────────────────────────────────────────────
    print(f"Загрузка данных из {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, sep=';', encoding='utf-8')
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {df.columns.tolist()}")

    # Проверяем наличие нужных колонок
    required_cols = ['generated_base', 'generated_sft', 'generated_ipo']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена в данных!")

    if 'category' not in df.columns:
        print("ВНИМАНИЕ: колонка 'category' не найдена — используем 'other' по умолчанию")
        df['category'] = 'other'

    # ── 2. Инициализация токенизатора ─────────────────────────────────────────
    print(f"\nИнициализация токенизатора для {JUDGE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

    # ── 3. Форматирование промтов (с перемешиванием) ──────────────────────────
    print("Подготовка промтов (перемешивание текстов)...")
    formatted_prompts, all_mappings = format_prompts_for_judge(df, tokenizer)

    # Отфильтровываем пустые
    valid_indices = [i for i, p in enumerate(formatted_prompts) if p.strip()]
    valid_prompts = [formatted_prompts[i] for i in valid_indices]
    print(f"Валидных записей для ранжирования: {len(valid_prompts)} из {len(formatted_prompts)}")

    # ── 4. Инициализация vLLM ─────────────────────────────────────────────────
    print(f"\nИнициализация vLLM с моделью {JUDGE_MODEL}...")
    llm = LLM(
        model=JUDGE_MODEL,
        max_model_len=4096 * 2,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.0,
    )

    # ── 5. Генерация ранжирований ─────────────────────────────────────────────
    print(f"\nНачало ранжирования {len(valid_prompts)} троек постов...")
    outputs = llm.generate(valid_prompts, sampling_params)

    # ── 6. Парсинг результатов ────────────────────────────────────────────────
    print("Парсинг ответов модели...")

    # Инициализируем колонки результатов
    df['rank_1'] = None       # Модель на 1-м месте (лучшая)
    df['rank_2'] = None       # Модель на 2-м месте
    df['rank_3'] = None       # Модель на 3-м месте (худшая)
    df['rank_reasoning'] = ""
    df['rank_raw_response'] = ""
    # Сохраняем маппинг перемешивания для воспроизводимости
    df['shuffle_mapping'] = ""

    parsed_count = 0
    failed_count = 0

    for idx, output in zip(valid_indices, outputs):
        response_text = output.outputs[0].text.strip()
        label_to_model = all_mappings[idx]

        result = parse_ranking_response(response_text, label_to_model)

        df.at[idx, 'rank_raw_response'] = result['raw_response']
        df.at[idx, 'shuffle_mapping'] = json.dumps(label_to_model, ensure_ascii=False)

        if result['parse_success']:
            parsed_count += 1
            df.at[idx, 'rank_1'] = result['rank_1']
            df.at[idx, 'rank_2'] = result['rank_2']
            df.at[idx, 'rank_3'] = result['rank_3']
            df.at[idx, 'rank_reasoning'] = result['reasoning']
        else:
            failed_count += 1

    print(f"\nУспешно распарсено: {parsed_count}")
    print(f"Не удалось распарсить: {failed_count}")

    # ── 7. Статистика: Win Rate ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СЛЕПОГО РАНЖИРОВАНИЯ (LLM-as-a-Judge)")
    print("=" * 60)

    ranked_df = df[df['rank_1'].notna()]
    total = len(ranked_df)

    if total > 0:
        print(f"\nВсего оценено троек: {total}\n")

        # Win rate (1-е место)
        print("Win Rate (1-е место):")
        for model in ['base', 'sft', 'ipo']:
            wins = (ranked_df['rank_1'] == model).sum()
            pct = wins / total * 100
            bar = '█' * int(pct / 2)
            print(f"  {model:6s}: {wins:5d} ({pct:5.1f}%) {bar}")

        # Средний ранг (1 = лучший, 3 = худший)
        print("\nСредний ранг (1=лучший, 3=худший):")
        for model in ['base', 'sft', 'ipo']:
            ranks = []
            for _, row in ranked_df.iterrows():
                if row['rank_1'] == model:
                    ranks.append(1)
                elif row['rank_2'] == model:
                    ranks.append(2)
                elif row['rank_3'] == model:
                    ranks.append(3)
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                print(f"  {model:6s}: {avg_rank:.2f}")

        # Win Rate по категориям
        if 'category' in df.columns:
            print("\nWin Rate (1-е место) по категориям:")
            print(f"  {'Категория':20s} | {'base':>8s} | {'sft':>8s} | {'ipo':>8s} | {'Всего':>6s}")
            print("  " + "-" * 60)
            for cat in sorted(ranked_df['category'].dropna().unique()):
                cat_df = ranked_df[ranked_df['category'] == cat]
                cat_total = len(cat_df)
                if cat_total == 0:
                    continue
                row_parts = [f"  {cat:20s}"]
                for model in ['base', 'sft', 'ipo']:
                    wins = (cat_df['rank_1'] == model).sum()
                    pct = wins / cat_total * 100
                    row_parts.append(f"{pct:6.1f}%")
                row_parts.append(f"{cat_total:6d}")
                print(" | ".join(row_parts))

    print("=" * 60)

    # ── 8. Сохранение ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, sep=';', index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в {OUTPUT_PATH}")
    print("Добавлены колонки: ['rank_1', 'rank_2', 'rank_3', "
          "'rank_reasoning', 'rank_raw_response', 'shuffle_mapping']")


if __name__ == "__main__":
    main()
