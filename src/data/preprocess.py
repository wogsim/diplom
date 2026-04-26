import os
import ast
import re
import pandas as pd
import numpy as np
import typer
from loguru import logger

app = typer.Typer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Удаляем URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text

def parse_reactions(reaction_str: str) -> int:
    """Парсит строку с реакциями (словарь) и возвращает сумму всех реакций."""
    if pd.isna(reaction_str):
        return 0
    try:
        # Если это строка, похожая на dict, парсим
        reactions_dict = ast.literal_eval(reaction_str)
        if isinstance(reactions_dict, dict):
            return sum(reactions_dict.values())
        return 0
    except (ValueError, SyntaxError):
        return 0

def calculate_z_score(group):
    mean_er = group.mean()
    std_er = group.std()
    
    # Защита от деления на ноль (если в канале все посты имеют одинаковый ER или пост всего один)
    if pd.isna(std_er) or std_er == 0:
        return pd.Series(0.0, index=group.index)
        
    return (group - mean_er) / std_er


def preprocess_data(input_path: str, output_path: str):
    logger.info(f"Загрузка сырых данных из {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Файл {input_path} не найден!")
        return
        
    df = pd.read_csv(input_path, sep=";")
    initial_len = len(df)
    logger.info(f"Исходное количество строк: {initial_len}")
    
    # 1. Фильтрация пустых текстов
    df = df.dropna(subset=['text']).copy()
    
    # Очистка текста
    logger.info("Очистка текста (удаление URL)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Удаление строк, где после очистки ничего не осталось
    df = df[df['clean_text'].str.len() > 10].copy()
    
    # 2. Вычисление метрик
    logger.info("Парсинг реакций и извлечение метрик...")
    df['total_reactions'] = df['reactions'].apply(parse_reactions)
    
    # Заполнение NaN нулями для числовых столбцов
    for col in ['views_count', 'forwards_count', 'replies_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Вычисление Reward (Engagement Rate)
    # ER = (Reactions + Forwards + Comments) / Views
    # Чтобы избежать деления на 0, добавим небольшой epsilon (или просто mask)
    total_engagements = df['total_reactions'] + df['forwards_count'] + df['replies_count']
    
    # Убираем посты, у которых почему-то 0 просмотров, чтобы не портить статистику
    df = df[df['views_count'] > 0].copy()
    
    df['base_er'] = total_engagements / df['views_count']
    
    # Применяем трансформацию с группировкой по компании/каналу
    df['reward'] = df.groupby('channel_handle')['base_er'].transform(calculate_z_score)

    # Клиппинг награды (Защита градиентов)
    # Ограничиваем награду диапазоном [-3, 3] или [-5, 5], 
    # чтобы один вирусный мемас не взорвал веса модели при обучении.
    df['reward'] = df['reward'].clip(-3.0, 3.0)
    
    # Создаем финальный датасет для обучения Reward Model (RM)
    # Для регрессионной RM нам нужны колонки: ['text', 'reward']
    final_cols = ['id', 'date', 'company_name', 'channel_handle', 'clean_text', 'views_count', 'total_reactions', 'forwards_count', 'replies_count', 'base_er', 'reward']
    df_final = df[final_cols].copy()
    
    # Сортировка по дате или ER (по желанию)
    df_final = df_final.sort_values(by='reward', ascending=False)
    
    df_final.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")
    
    final_len = len(df_final)
    logger.info(f"Предобработка завершена. Оставлено постов: {final_len} (отфильтровано {initial_len - final_len}).")
    logger.info(f"Файл сохранен в: {output_path}")

@app.command()
def main(
    input_file: str = typer.Option("Data/raw_posts.csv", help="Путь к сырому датасету"),
    output_file: str = typer.Option("Data/processed_posts.csv", help="Путь для сохранения обработанного датасета")
):
    preprocess_data(input_file, output_file)

if __name__ == "__main__":
    app()
