"""
Оценка сгенерированных постов с помощью LLM (слепая оценка с якорем).

Для каждого промта берёт три генерации (base, sft, ipo), сравнивает их
с оригинальным текстом (ground truth). Использует все 6 перестановок
для устранения позиционного смещения.

Модель-судья оценивает фактологическую точность и качество текста.

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

# Все модели
MODELS = ["base", "sft", "ipo"]

# Все 6 перестановок троек моделей
ALL_PERMUTATIONS = list(permutations(MODELS))  # 3! = 6

# ─── Промт для оценки ────────────────────────────────────────────────────────

EVAL_PROMPT_TEMPLATE = """Ты — главный редактор ИТ-медиа. Твоя задача — выбрать лучший вариант Telegram-поста, сравнив три генерации с исходным текстом.

ИСХОДНЫЙ ТЕКСТ (Якорь):
{ground_truth_text}

ВАРИАНТЫ ДЛЯ СРАВНЕНИЯ:
[Вариант 1]: {text_1}

[Вариант 2]: {text_2}

[Вариант 3]: {text_3}

ИНСТРУКЦИЯ ПО ОЦЕНКЕ:
1. Фактологическая точность (Anchor Check): Текст должен сохранить все ключевые факты, цифры и ИТ-термины из Исходного текста. Штрафуй за потерю конкретики.
2. Отсутствие "воды": Штрафуй за бессодержательные клише ("высокие стандарты", "инновационный прорыв"), которых нет в Исходном тексте.
3. Доменная адаптация: Текст должен выглядеть как профессиональный пост для B2B ИТ-рынка (структура, читаемость, уместный стиль).
4. Качество выше оригинала: Если генерация исправила косноязычие оригинала, сохранив смысл — это плюс. Если генерация размыла смысл ради красоты — это минус.

Твой ответ должен быть строго в формате JSON:
{{
  "winner": "Вариант X",
  "reasoning": "Четкое обоснование: что победитель сохранил из оригинала, и почему проигравшие (особенно IPO/SFT версии) не справились (например, добавили воды или потеряли факты).",
  "fact_retention_score": 0,
  "writing_quality_score": 0
}}"""


def build_eval_prompt(ground_truth: str, perm: tuple, texts: dict[str, str]) -> str:
    """
    Собирает промт для одной перестановки.

    perm — кортеж из 3 ключей моделей, например ("sft", "base", "ipo")
    texts — dict {"base": "...", "sft": "...", "ipo": "..."}
    """
    return EVAL_PROMPT_TEMPLATE.format(
        ground_truth_text=ground_truth,
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
        ground_truth = str(row.get('real_post', '')).strip()
        base_text = str(row.get('generated_base', '')).strip()
        sft_text = str(row.get('generated_sft', '')).strip()
        ipo_text = str(row.get('generated_ipo', '')).strip()

        # Пропускаем строки, где хотя бы один текст пустой
        if not ground_truth or not base_text or not sft_text or not ipo_text:
            continue

        texts = {"base": base_text, "sft": sft_text, "ipo": ipo_text}

        # Генерируем промт для каждой из 6 перестановок
        for perm in ALL_PERMUTATIONS:
            # perm = (model_for_variant1, model_for_variant2, model_for_variant3)
            label_to_model = {
                "Вариант 1": perm[0],
                "Вариант 2": perm[1],
                "Вариант 3": perm[2],
            }

            user_message = build_eval_prompt(ground_truth, perm, texts)
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


def parse_eval_response(
    response: str,
    label_to_model: dict[str, str],
) -> dict | None:
    """
    Извлекает результат оценки из JSON-ответа и маппит обратно на модели.

    Возвращает dict с ключами:
      winner — имя модели (base/sft/ipo)
      fact_retention_score — оценка 1-10
      writing_quality_score — оценка 1-10
      reasoning — обоснование
    или None при ошибке парсинга.
    """
    if not response or not response.strip():
        return None

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())

        # Извлекаем победителя
        winner_label = parsed.get("winner", "").strip()
        # Нормализация: "Вариант 1", "вариант 1", "Variant 1" и т.д.
        variant_num = re.search(r'(\d)', winner_label)
        if not variant_num:
            return None

        normalized_label = f"Вариант {variant_num.group(1)}"
        winner_model = label_to_model.get(normalized_label)
        if winner_model is None:
            return None

        # Извлекаем оценки
        fact_score = parsed.get("fact_retention_score")
        writing_score = parsed.get("writing_quality_score")

        if fact_score is not None:
            fact_score = max(1, min(10, int(fact_score)))
        if writing_score is not None:
            writing_score = max(1, min(10, int(writing_score)))

        reasoning = str(parsed.get("reasoning", "")).strip()

        return {
            "winner": winner_model,
            "fact_retention_score": fact_score,
            "writing_quality_score": writing_score,
            "reasoning": reasoning,
        }

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def aggregate_results(row_results: list[dict]) -> dict:
    """
    Агрегирует результаты всех перестановок для одной строки.

    row_results — список dict'ов из parse_eval_response

    Возвращает:
    - win_count: кол-во побед каждой модели
    - avg_fact_score / avg_writing_score: средние оценки победителя
    - best_model: модель с наибольшим числом побед
    - num_valid: сколько перестановок удалось распарсить
    """
    win_counts = {"base": 0, "sft": 0, "ipo": 0}
    fact_scores = {"base": [], "sft": [], "ipo": []}
    writing_scores = {"base": [], "sft": [], "ipo": []}

    for result in row_results:
        winner = result["winner"]
        win_counts[winner] += 1

        if result["fact_retention_score"] is not None:
            fact_scores[winner].append(result["fact_retention_score"])
        if result["writing_quality_score"] is not None:
            writing_scores[winner].append(result["writing_quality_score"])

    num_valid = len(row_results)

    # Лучшая модель = больше всего побед; при равенстве — первая в порядке
    best_model = max(MODELS, key=lambda m: win_counts[m])

    # Средние оценки (только для побед данной модели)
    avg_fact = {}
    avg_writing = {}
    for model in MODELS:
        avg_fact[model] = (
            sum(fact_scores[model]) / len(fact_scores[model])
            if fact_scores[model] else None
        )
        avg_writing[model] = (
            sum(writing_scores[model]) / len(writing_scores[model])
            if writing_scores[model] else None
        )

    return {
        "wins_base": win_counts["base"],
        "wins_sft": win_counts["sft"],
        "wins_ipo": win_counts["ipo"],
        "avg_fact_base": avg_fact["base"],
        "avg_fact_sft": avg_fact["sft"],
        "avg_fact_ipo": avg_fact["ipo"],
        "avg_writing_base": avg_writing["base"],
        "avg_writing_sft": avg_writing["sft"],
        "avg_writing_ipo": avg_writing["ipo"],
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
    required_cols = ['real_post', 'generated_base', 'generated_sft', 'generated_ipo']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена в данных!")

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
        max_model_len=4096 * 8,
        gpu_memory_utilization=0.97,
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

    # ── 5. Генерация оценок (один большой батч) ──────────────────────────────
    print(f"\nНачало оценки ({len(all_prompts)} запросов)...")
    outputs = llm.generate(all_prompts, sampling_params)

    # ── 6. Парсинг и группировка по строкам ───────────────────────────────────
    print("Парсинг ответов модели...")

    from collections import defaultdict
    row_results = defaultdict(list)

    total_parsed = 0
    total_failed = 0

    for meta, output in zip(all_meta, outputs):
        response_text = output.outputs[0].text.strip()
        label_to_model = meta["label_to_model"]
        row_idx = meta["row_idx"]

        result = parse_eval_response(response_text, label_to_model)
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

    result_cols = [
        'wins_base', 'wins_sft', 'wins_ipo',
        'avg_fact_base', 'avg_fact_sft', 'avg_fact_ipo',
        'avg_writing_base', 'avg_writing_sft', 'avg_writing_ipo',
        'best_model', 'num_valid_perms',
    ]
    for col in result_cols:
        df[col] = None

    rows_with_results = 0

    for row_idx, results in row_results.items():
        if not results:
            continue

        agg = aggregate_results(results)
        rows_with_results += 1

        for col in result_cols:
            df.at[row_idx, col] = agg[col]

    print(f"Строк с результатами: {rows_with_results}")

    # ── 8. Статистика ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СЛЕПОЙ ОЦЕНКИ (LLM-as-a-Judge с якорем)")
    print(f"Все 6 перестановок, агрегация по {rows_with_results} строкам")
    print("=" * 60)

    ranked_df = df[df['best_model'].notna()]
    total = len(ranked_df)

    if total > 0:
        # Общий Win Rate (best_model — победитель агрегации)
        print(f"\nОбщий Win Rate (лучшая модель по числу побед):")
        for model in MODELS:
            wins = (ranked_df['best_model'] == model).sum()
            pct = wins / total * 100
            bar = '█' * int(pct / 2)
            print(f"  {model:6s}: {wins:5d} ({pct:5.1f}%) {bar}")

        # Суммарные победы по всем перестановкам
        print(f"\nСуммарные победы (из {total * 6} макс. голосов):")
        for model in MODELS:
            col = f'wins_{model}'
            total_wins = ranked_df[col].astype(int).sum()
            max_possible = total * 6
            pct = total_wins / max_possible * 100
            bar = '█' * int(pct / 2)
            print(f"  {model:6s}: {total_wins:5d} ({pct:5.1f}%) {bar}")

        # Средние оценки fact_retention
        print(f"\nСредний Fact Retention Score (при победах модели):")
        for model in MODELS:
            col = f'avg_fact_{model}'
            valid = ranked_df[col].dropna().astype(float)
            if len(valid) > 0:
                print(f"  {model:6s}: {valid.mean():.2f}/10  (n={len(valid)})")
            else:
                print(f"  {model:6s}: —")

        # Средние оценки writing_quality
        print(f"\nСредний Writing Quality Score (при победах модели):")
        for model in MODELS:
            col = f'avg_writing_{model}'
            valid = ranked_df[col].dropna().astype(float)
            if len(valid) > 0:
                print(f"  {model:6s}: {valid.mean():.2f}/10  (n={len(valid)})")
            else:
                print(f"  {model:6s}: —")

        # Среднее кол-во успешных парсингов
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
