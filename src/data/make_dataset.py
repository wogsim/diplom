import asyncio
import pandas as pd
from dotenv import load_dotenv
import os
from loguru import logger
import typer

# Используем абсолютный импорт от src.data.collector, если запускаем как python -m src.data.make_dataset
# Или просто импортируем локально
try:
    from src.data.collector import TelegramChanelParser
except ModuleNotFoundError:
    from collector import TelegramChanelParser

app = typer.Typer()

async def collect_telegram_data(limit: int = 500, max_channels: int = 0):
    """
    Асинхронный пайплайн для сбора постов B2B компаний.
    limit: Максимальное количество последних постов для каждого канала (0 - без лимита).
    max_channels: Максимальное количество каналов для парсинга за один запуск (для тестов). 0 - все.
    """
    env_path = os.path.join("src", "data", ".env")
    load_dotenv(env_path)
    
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")
    
    if not api_id or not api_hash:
        logger.error(f"TG_API_ID и TG_API_HASH не найдены в {env_path}")
        return

    companies_csv = os.path.join("Data", "list_of_companies.csv")
    if not os.path.exists(companies_csv):
        logger.error(f"Файл {companies_csv} не найден.")
        return

    df_companies = pd.read_csv(companies_csv, sep=";")
    
    parser = TelegramChanelParser(api_id=api_id, api_hash=api_hash, logger=logger)
    
    session_path = os.path.join("src", "my_session")
    await parser.connect_client(session_name=session_path)
    
    out_path = os.path.join("Data", "raw_posts.csv")
    processed_channels = set()
    
    # Загружаем уже спарсенные каналы, чтобы продолжить с места остановки
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path, sep=";")
            if "channel_handle" in existing_df.columns:
                processed_channels = set(existing_df['channel_handle'].dropna().astype(str).unique())
                logger.info(f"Найдено {len(processed_channels)} уже спарсенных каналов. Они будут пропущены.")
        except Exception as e:
            logger.warning(f"Не удалось прочитать {out_path}: {e}")
    
    channels_processed_in_run = 0
    
    try:
        for idx, row in df_companies.iterrows():
            tg_handle = str(row['tg']).strip()
            name = str(row['name']).strip()
            
            if pd.isna(tg_handle) or tg_handle.lower() == 'н/д' or tg_handle == 'nan' or tg_handle == 'None':
                 logger.warning(f"Пропуск компании {name}, нет TG")
                 continue
                 
            if tg_handle.startswith('@'):
                tg_handle = tg_handle[1:]
                
            if tg_handle in processed_channels:
                logger.info(f"Пропуск {name} (@{tg_handle}): канал уже был спарсен ранее.")
                continue
                
            logger.info(f"Сбор постов для {name} / @{tg_handle}")
            
            try:
                kwargs = {}
                if limit > 0:
                    kwargs['limit'] = limit
                    
                posts_df = await parser.fetch_posts(channel=tg_handle, dataframe=True, **kwargs)
                
                if not posts_df.empty:
                    posts_df['company_name'] = name
                    posts_df['channel_handle'] = tg_handle
                    
                    # Сразу сохраняем данные этого канала, чтобы не потерять прогресс
                    header = not os.path.exists(out_path)
                    posts_df.to_csv(out_path, index=False, sep=";", encoding="utf-8-sig", mode='a', header=header)
                    
                    logger.info(f"-> Успешно скачано и сохранено {len(posts_df)} постов.")
                else:
                    logger.warning(f"-> Канал {tg_handle} пуст или нет доступа.")
                    
                processed_channels.add(tg_handle)
                channels_processed_in_run += 1
                
                if max_channels > 0 and channels_processed_in_run >= max_channels:
                    logger.info(f"Достигнут лимит в {max_channels} каналов (режим тестирования). Остановка.")
                    break
                    
                logger.info("Ждем 5 секунд перед следующим каналом...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Ошибка при сборе {name} (@{tg_handle}): {e}")
                
    finally:
        await parser.disconnect_client()
        logger.info("Парсинг завершен или прерван.")

@app.command()
def main(
    limit: int = typer.Option(500, help="Количество последних постов с каждого канала. 0 - выгрузить все."),
    max_channels: int = typer.Option(0, help="Ограничение на количество каналов за этот запуск. 0 - все из CSV.")
):
    logger.info("Начинаем процесс сборки данных из Telegram.")
    asyncio.run(collect_telegram_data(limit=limit, max_channels=max_channels))

if __name__ == "__main__":
    app()
