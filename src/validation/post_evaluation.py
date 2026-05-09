"""
Ранжирование сгенерированных постов с помощью LLM (слепая оценка).

Для каждого промта берёт три генерации (base, sft, ipo) и создаёт ВСЕ 6
возможных перестановок (3! = 6), присваивая временные ID «Текст 1/2/3».
Это устраняет позиционное смещение (position bias) модели-судьи.

Результаты агрегируются: для каждой строки подсчитывается средний ранг
каждой модели по всем 6 перестановкам.

Предназначен для запуска в Google Colab с GPU.
"""

import pandas as pd
import json
import re
from itertools import permutations
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ─── Настройки ────────────────────────────────────────────────────────────────

# Входной файл — результат post_classification.py (содержит колонку category)
INPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison_classified.csv'
OUTPUT_PATH = '../drive/MyDrive/diplom/datasets/evaluation_comparison_ranked.csv'

# Модель-судья (отличается от моделей-генераторов для объективности)
JUDGE_MODEL = "google/gemma-4-26B-A4B-it"

# Маппинг категорий на человекочитаемые названия
CATEGORY_NAMES = {
    "tech_guide": "Технический гайд",
    "case_study": "Кейс/Внедрение",
    "product_update": "Продукт/Анонс",
    "market_news": "Новости/Аналитика",
    "expert_opinion": "Мнение эксперта",
    "other": "Другое",
}

# Все модели
MODELS = ["base", "sft", "ipo"]

# Все 6 перестановок троек моделей
ALL_PERMUTATIONS = list(permutations(MODELS))  # 3! = 6

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


def build_ranking_prompt(category: str, perm: tuple, texts: dict[str, str]) -> str:
    """
    Собирает промт для одной перестановки.

    perm — кортеж из 3 ключей моделей, например ("sft", "base", "ipo")
    texts — dict {"base": "...", "sft": "...", "ipo": "..."}
    """
    category_name = CATEGORY_NAMES.get(category, category or "Неизвестная")

    return RANKING_PROMPT_TEMPLATE.format(
        category_name=category_name,
        text_1=texts[perm[0]],
        text_2=texts[perm[1]],
        text_3=texts[perm[2]],
    )


def prepare_all_prompts(
    df: pd.DataFrame,
    tokenizer,
) -> tuple[list[str], list[dict]]:
    """
    Для каждой строки DataFrame генерирует 6 промтов (все перестановки).

    Возвращает:
    - all_prompts: плоский список отформатированных промтов
    - all_meta: список метаданных для каждого промта
      (row_idx, perm, label_to_model)
    """
    all_prompts = []
    all_meta = []

    for row_idx, row in df.iterrows():
        base_text = str(row.get('generated_base', '')).strip()
        sft_text = str(row.get('generated_sft', '')).strip()
        ipo_text = str(row.get('generated_ipo', '')).strip()
        category = str(row.get('category', 'other'))

        # Пропускаем строки, где хотя бы одна генерация пустая
        if not base_text or not sft_text or not ipo_text:
            continue

        texts = {"base": base_text, "sft": sft_text, "ipo": ipo_text}

        # Генерируем промт для каждой из 6 перестановок
        for perm in ALL_PERMUTATIONS:
            # perm = (model_for_text1, model_for_text2, model_for_text3)
            label_to_model = {
                "Текст 1": perm[0],
                "Текст 2": perm[1],
                "Текст 3": perm[2],
            }

            user_message = build_ranking_prompt(category, perm, texts)
            messages = [{"role": "user", "content": user_message}]

            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            all_prompts.append(chat_prompt)
            all_meta.append({
                "row_idx": row_idx,
                "perm": perm,
                "label_to_model": label_to_model,
            })

    return all_prompts, all_meta


def parse_ranking_response(
    response: str,
    label_to_model: dict[str, str],
) -> dict | None:
    """
    Извлекает ранжирование из JSON-ответа и маппит обратно на модели.
    Возвращает dict {"rank_1": model, "rank_2": model, "rank_3": model}
    или None при ошибке парсинга.
    """
    if not response or not response.strip():
        return None

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())

        places = {}
        for place_key, rank_key in [("1st_place", "rank_1"),
                                     ("2nd_place", "rank_2"),
                                     ("3rd_place", "rank_3")]:
            label = parsed.get(place_key, "").strip()
            text_num = re.search(r'(\d)', label)
            if text_num:
                normalized_label = f"Текст {text_num.group(1)}"
                model = label_to_model.get(normalized_label)
                places[rank_key] = model
            else:
                return None

        models = [places["rank_1"], places["rank_2"], places["rank_3"]]
        if None in models or len(set(models)) != 3:
            return None

        return places

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def aggregate_rankings(row_results: list[dict]) -> dict:
    """
    Агрегирует результаты всех перестановок для одной строки.

    row_results — список dict'ов {"rank_1": model, "rank_2": model, "rank_3": model}

    Возвращает:
    - avg_rank: средний ранг каждой модели
    - win_count: кол-во 1-х мест
    - best_model: модель с наименьшим средним рангом
    - num_valid: сколько перестановок удалось распарсить
    """
    rank_sums = {"base": 0, "sft": 0, "ipo": 0}
    rank_counts = {"base": 0, "sft": 0, "ipo": 0}
    win_counts = {"base": 0, "sft": 0, "ipo": 0}

    for result in row_results:
        for rank_key, rank_val in [("rank_1", 1), ("rank_2", 2), ("rank_3", 3)]:
            model = result[rank_key]
            rank_sums[model] += rank_val
            rank_counts[model] += 1
            if rank_val == 1:
                win_counts[model] += 1

    num_valid = len(row_results)
    avg_ranks = {}
    for model in MODELS:
        if rank_counts[model] > 0:
            avg_ranks[model] = rank_sums[model] / rank_counts[model]
        else:
            avg_ranks[model] = None

    # Лучшая модель = наименьший средний ранг
    best_model = min(
        (m for m in MODELS if avg_ranks[m] is not None),
        key=lambda m: avg_ranks[m],
        default=None,
    )

    return {
        "avg_rank_base": avg_ranks["base"],
        "avg_rank_sft": avg_ranks["sft"],
        "avg_rank_ipo": avg_ranks["ipo"],
        "wins_base": win_counts["base"],
        "wins_sft": win_counts["sft"],
        "wins_ipo": win_counts["ipo"],
        "best_model": best_model,
        "num_valid_perms": num_valid,
    }


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

    # ── 3. Подготовка промтов (все 6 перестановок на строку) ───────────────────
    print("Подготовка промтов (6 перестановок на каждую тройку)...")
    all_prompts, all_meta = prepare_all_prompts(df, tokenizer)
    print(f"Всего промтов для модели: {len(all_prompts)} "
          f"(~{len(all_prompts) // 6} строк × 6 перестановок)")

    # ── 4. Инициализация vLLM ─────────────────────────────────────────────────
    print(f"\nИнициализация vLLM с моделью {JUDGE_MODEL}...")
    llm = LLM(
        model=JUDGE_MODEL,
        max_model_len=4096 * 4,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        #quantization='bitsandbytes',     # 4-bit квантизация (NF4)
        #load_format='bitsandbytes',
    )

    sampling_params = SamplingParams(
        temperature=1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.0,
    )

    # ── 5. Генерация ранжирований (один большой батч) ─────────────────────────
    print(f"\nНачало ранжирования ({len(all_prompts)} запросов)...")
    outputs = llm.generate(all_prompts, sampling_params)

    # ── 6. Парсинг и группировка по строкам ───────────────────────────────────
    print("Парсинг ответов модели...")

    # Группируем результаты по row_idx
    from collections import defaultdict
    row_results = defaultdict(list)

    total_parsed = 0
    total_failed = 0

    for meta, output in zip(all_meta, outputs):
        response_text = output.outputs[0].text.strip()
        label_to_model = meta["label_to_model"]
        row_idx = meta["row_idx"]

        result = parse_ranking_response(response_text, label_to_model)
        if result is not None:
            row_results[row_idx].append(result)
            total_parsed += 1
        else:
            total_failed += 1

    print(f"\nВсего ответов: {len(all_meta)}")
    print(f"Успешно распарсено: {total_parsed} ({total_parsed / len(all_meta) * 100:.1f}%)")
    print(f"Не удалось распарсить: {total_failed}")

    # ── 7. Агрегация результатов ──────────────────────────────────────────────
    print("Агрегация результатов по всем перестановкам...")

    # Инициализируем колонки
    result_cols = [
        'avg_rank_base', 'avg_rank_sft', 'avg_rank_ipo',
        'wins_base', 'wins_sft', 'wins_ipo',
        'best_model', 'num_valid_perms',
    ]
    for col in result_cols:
        df[col] = None

    rows_with_results = 0

    for row_idx, results in row_results.items():
        if not results:
            continue

        agg = aggregate_rankings(results)
        rows_with_results += 1

        for col in result_cols:
            df.at[row_idx, col] = agg[col]

    print(f"Строк с результатами: {rows_with_results}")

    # ── 8. Статистика ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СЛЕПОГО РАНЖИРОВАНИЯ (LLM-as-a-Judge)")
    print(f"Все 6 перестановок, агрегация по {rows_with_results} строкам")
    print("=" * 60)

    ranked_df = df[df['best_model'].notna()]
    total = len(ranked_df)

    if total > 0:
        # Общий Win Rate (по best_model — победитель агрегации)
        print(f"\nОбщий Win Rate (лучшая модель по среднему рангу):")
        for model in MODELS:
            wins = (ranked_df['best_model'] == model).sum()
            pct = wins / total * 100
            bar = '█' * int(pct / 2)
            print(f"  {model:6s}: {wins:5d} ({pct:5.1f}%) {bar}")

        # Средний ранг по всему датасету
        print(f"\nСредний ранг (1=лучший, 3=худший):")
        for model in MODELS:
            col = f'avg_rank_{model}'
            mean_rank = ranked_df[col].astype(float).mean()
            print(f"  {model:6s}: {mean_rank:.3f}")

        # Суммарные 1-е места по всем перестановкам
        print(f"\nСуммарные 1-е места (из {total * 6} макс. голосов):")
        for model in MODELS:
            col = f'wins_{model}'
            total_wins = ranked_df[col].astype(int).sum()
            max_possible = total * 6
            pct = total_wins / max_possible * 100
            bar = '█' * int(pct / 2)
            print(f"  {model:6s}: {total_wins:5d} ({pct:5.1f}%) {bar}")

        # Среднее кол-во успешных парсингов на строку
        avg_valid = ranked_df['num_valid_perms'].astype(float).mean()
        print(f"\nСреднее кол-во валидных перестановок на строку: {avg_valid:.1f}/6")

        # Win Rate по категориям
        if 'category' in df.columns:
            print(f"\nWin Rate (best_model) по категориям:")
            print(f"  {'Категория':20s} | {'base':>8s} | {'sft':>8s} | {'ipo':>8s} | {'Всего':>6s}")
            print("  " + "-" * 60)
            for cat in sorted(ranked_df['category'].dropna().unique()):
                cat_df = ranked_df[ranked_df['category'] == cat]
                cat_total = len(cat_df)
                if cat_total == 0:
                    continue
                row_parts = [f"  {cat:20s}"]
                for model in MODELS:
                    wins = (cat_df['best_model'] == model).sum()
                    pct = wins / cat_total * 100
                    row_parts.append(f"{pct:6.1f}%")
                row_parts.append(f"{cat_total:6d}")
                print(" | ".join(row_parts))

    print("=" * 60)

    # ── 9. Сохранение ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, sep=';', index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в {OUTPUT_PATH}")
    print(f"Добавлены колонки: {result_cols}")


if __name__ == "__main__":
    main()
