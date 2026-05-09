import pandas as pd
from pathlib import Path

def main():
    # Настройки путей
    TEST_DATA_PATH = '../drive/MyDrive/diplom/datasets/processed_posts_with_prompts_test.csv'
    BASE_RESULTS = '../drive/MyDrive/diplom/datasets/generated_base.csv'
    SFT_RESULTS = '../drive/MyDrive/diplom/datasets/generated_sft.csv'
    IPO_RESULTS = '../drive/MyDrive/diplom/datasets/generated_ipo.csv'
    
    FINAL_OUTPUT = '../drive/MyDrive/diplom/datasets/evaluation_comparison.csv'

    print("Чтение исходного тестового датасета...")
    df_test = pd.read_csv(TEST_DATA_PATH, sep=";")
    
    try:
        df_base = pd.read_csv(BASE_RESULTS, sep=";")
        df_test['generated_base'] = df_base['generated_base']
        print("Базовые генерации успешно добавлены.")
    except Exception as e:
        print(f"Не удалось загрузить генерации базовой модели: {e}")
        
    try:
        df_sft = pd.read_csv(SFT_RESULTS, sep=";")
        df_test['generated_sft'] = df_sft['generated_sft']
        print("SFT генерации успешно добавлены.")
    except Exception as e:
        print(f"Не удалось загрузить генерации SFT модели: {e}")

    try:
        df_ipo = pd.read_csv(IPO_RESULTS, sep=";")
        df_test['generated_ipo'] = df_ipo['generated_ipo']
        print("IPO генерации успешно добавлены.")
    except Exception as e:
        print(f"Не удалось загрузить генерации IPO модели: {e}")

    # Оставляем только нужные колонки для сравнения
    cols_to_keep = ['promt', 'clean_text']
    
    for col in ['generated_base', 'generated_sft', 'generated_ipo']:
        if col in df_test.columns:
            cols_to_keep.append(col)
            
    df_final = df_test[cols_to_keep].copy()
    
    # Переименовываем clean_text в real_post для понятности
    df_final = df_final.rename(columns={'clean_text': 'real_post'})

    df_final.to_csv(FINAL_OUTPUT, sep=";", index=False)
    print(f"\nГотово! Итоговая таблица со всеми генерациями сохранена в {FINAL_OUTPUT}")
    print("Теперь вы можете визуально сравнивать тексты или запустить подсчет метрик!")

if __name__ == "__main__":
    main()
