"""
Подготовка датасетов для дообучения Gemma 4 E2B IT.

Создаёт два датасета из processed_posts_with_prompts.csv:
1. SFT-датасет (conversational формат для SFTTrainer)
2. IPO-датасет (preference формат для DPOTrainer с loss_type="ipo")
"""

import pandas as pd
import numpy as np
import json
import os
import torch
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer, util


# ── Пути ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
INPUT_FILE = DATA_DIR / "processed_posts_with_prompts_train.csv"
SFT_OUTPUT = DATA_DIR / "sft_dataset.jsonl"
IPO_OUTPUT = DATA_DIR / "ipo_dataset.jsonl"



# ── Загрузка и очистка ──────────────────────────────────────────────────
def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Загружает CSV, удаляет строки с пустыми promt / clean_text."""
    logger.info(f"Загрузка данных из {path}")
    df = pd.read_csv(path, sep=";")
    initial_len = len(df)

    df = df.dropna(subset=["promt", "clean_text"])
    df = df[df["promt"].str.strip().astype(bool) & df["clean_text"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    logger.info(f"Загружено {initial_len} строк, после очистки: {len(df)}")
    return df



# ── SFT-датасет ──────────────────────────────────────────────────────────
def build_sft_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Формирует SFT-датасет берем посты где reward > 0.4 в conversational формате (JSONL):
    {"prompt": [{"role":"user","content":"..."}],
     "completion": [{"role":"assistant","content":"..."}]}
    """
    logger.info("Формирование SFT-датасета...")
    records = []

    df = df[df['reward'] > 0.4]

    for _, row in df.iterrows():
        record = {
            "prompt": [{"role": "user", "content": row["promt"]}],
            "completion": [{"role": "assistant", "content": row["clean_text"]}],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"SFT-датасет сохранён: {output_path}  ({len(records)} примеров)")


# ── IPO-датасет ──────────────────────────────────────────────────────────
def build_ipo_dataset(
    df: pd.DataFrame,
    output_path: Path,
    top_quantile: float = 0.75,
    bottom_quantile: float = 0.25,
    max_pairs_per_prompt: int = 3,
) -> None:
    """
    Формирует IPO (preference) датасет (JSONL):
    {"prompt": [{"role":"user","content":"..."}],
     "chosen": [{"role":"assistant","content":"<high-engagement text>"}],
     "rejected": [{"role":"assistant","content":"<low-engagement text>"}]}

    Стратегия построения пар:
    1. Вычисляем engagement score для каждого поста.
    2. Делим на «хорошие» (≥ top_quantile) и «плохие» (≤ bottom_quantile).
    3. Внутри каждого канала (channel_handle) формируем пары chosen/rejected.
    4. Если каналов мало — дополнительно создаём cross-channel пары.
    """
    logger.info("Формирование IPO-датасета...")

    df = df.copy()

    top_thresh = df["reward"].quantile(top_quantile)
    bot_thresh = df["reward"].quantile(bottom_quantile)

    good = df[df["reward"] >= top_thresh]
    bad = df[df["reward"] <= bot_thresh]

    logger.info(
        f"Порог хороших: {top_thresh:.4f} ({len(good)} постов), "
        f"порог плохих: {bot_thresh:.4f} ({len(bad)} постов)"
    )

    pairs = []

    logger.info("Загрузка модели SentenceTransformers (cointegrated/rubert-tiny2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("cointegrated/rubert-tiny2", device=device)

    # ── Внутриканальные пары ──
    channels = set(good["channel_handle"].unique()) & set(bad["channel_handle"].unique())
    for ch in channels:
        ch_good = good[good["channel_handle"] == ch]
        ch_bad = bad[bad["channel_handle"] == ch]

        if len(ch_bad) == 0 or len(ch_good) == 0:
            continue
            
        # Векторизация постов канала
        good_vecs = embedder.encode(ch_good["clean_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
        bad_vecs = embedder.encode(ch_bad["clean_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
        
        sim_matrix = util.cos_sim(good_vecs, bad_vecs)

        for i, (_, g_row) in enumerate(ch_good.iterrows()):
            scores = sim_matrix[i]
            # Сортируем индексы плохих постов по убыванию сходства
            top_indices = torch.argsort(scores, descending=True)[:max_pairs_per_prompt]
            for idx in top_indices:
                b_row = ch_bad.iloc[idx.item()]
                sim_score = round(scores[idx].item(), 4)
                pairs.append(_make_preference_record(g_row, b_row, similarity=sim_score))

    # ── Кросс-канальные пары (дополнение) ──
    if len(pairs) < 500:
        logger.info("Мало внутриканальных пар, добавляем кросс-канальные с учетом семантического сходства...")
        n_extra = min(2000, len(good))
        
        sample_good = good.sample(n_extra, random_state=42) if len(good) > n_extra else good
        sample_bad = bad.sample(min(n_extra * 3, len(bad)), random_state=42) if len(bad) > n_extra * 3 else bad
        
        if len(sample_bad) > 0 and len(sample_good) > 0:
            good_vecs = embedder.encode(sample_good["clean_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
            bad_vecs = embedder.encode(sample_bad["clean_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
            
            sim_matrix = util.cos_sim(good_vecs, bad_vecs)
            
            for i, (_, g_row) in enumerate(sample_good.iterrows()):
                best_idx = torch.argsort(sim_matrix[i], descending=True)[0]
                b_row = sample_bad.iloc[best_idx.item()]
                sim_score = round(sim_matrix[i][best_idx].item(), 4)
                pairs.append(_make_preference_record(g_row, b_row, similarity=sim_score))

    # Убираем дубликаты
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p["chosen"][0]["content"][:100], p["rejected"][0]["content"][:100])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in unique_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"IPO-датасет сохранён: {output_path}  ({len(unique_pairs)} пар)")


def _make_preference_record(good_row: pd.Series, bad_row: pd.Series, similarity: float = None) -> dict:
    """Создаёт одну запись preference-датасета."""
    # Промпт берём от «хорошего» поста
    record = {
        "prompt": [{"role": "user", "content": good_row["promt"]}],
        "chosen": [{"role": "assistant", "content": good_row["clean_text"]}],
        "rejected": [{"role": "assistant", "content": bad_row["clean_text"]}],
    }
    if similarity is not None:
        record["similarity_score"] = similarity
    return record


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    df = load_and_clean(INPUT_FILE)

    build_sft_dataset(df, SFT_OUTPUT)
    build_ipo_dataset(df, IPO_OUTPUT)

    logger.success("Все датасеты сформированы!")


if __name__ == "__main__":
    main()
