import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

def main():
    # Настройки путей (адаптируйте под свой Google Drive)
    TEST_DATA_PATH = '../drive/MyDrive/Диплом/datasets/processed_posts_with_prompts_test.csv'
    OUTPUT_PATH = '../drive/MyDrive/Диплом/datasets/generated_base.csv'
    
    # Имя базовой модели
    MODEL_PATH = "google/gemma-4-E2B-it" # Или другая ваша базовая модель (например, google/gemma-2-2b-it)

    print(f"Загрузка тестовых данных из {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, sep=";")
    
    # Для теста можно взять часть (например первые 1000 строк) чтобы не ждать долго
    # df = df.head(1000)

    prompts = df['promt'].fillna("").tolist()
    
    print(f"Инициализация токенизатора для {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Форматируем промпты под формат чата модели (Gemma)
    print("Применение chat_template...")
    formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    print(f"Инициализация vLLM с моделью {MODEL_PATH}...")
    # Настройки vLLM (max_model_len ограничиваем, чтобы экономить VRAM)
    llm = LLM(
        model=MODEL_PATH, 
        max_model_len=4096*2, 
        gpu_memory_utilization=0.90, # Используем 90% памяти видеокарты
        tensor_parallel_size=1 # Измените на количество GPU, если их несколько
    )
    
    # Настройки генерации
    sampling_params = SamplingParams(
        temperature=1, 
        top_p=0.95, 
        max_tokens=1024,
        repetition_penalty=1.1
    )

    print("Начало генерации...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Извлекаем сгенерированный текст
    generated_texts = [out.outputs[0].text.strip() for out in outputs]
    df['generated_base'] = generated_texts
    
    # Сохраняем результат
    df.to_csv(OUTPUT_PATH, sep=";", index=False)
    print(f"Готово! Результаты сохранены в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
