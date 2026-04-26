import pandas as pd
from vllm import LLM, SamplingParams
import time
import os

def load_data(input_file: str) -> pd.DataFrame:
    """Загружает данные и возвращает DataFrame."""
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден. Проверьте путь.")
        # Fallback для Colab
        if os.path.exists('processed_posts.csv'):
            print("Найден локальный файл processed_posts.csv")
            input_file = 'processed_posts.csv'
            
    print(f"Загрузка данных из {input_file}...")
    df = pd.read_csv(input_file, sep=';')
    return df

def format_prompts(texts: list, system_prompt: str) -> list:
    """Форматирует тексты в промты для модели (формат Gemma)."""
    formatted_prompts = []
    for text in texts:
        if not str(text).strip():
            formatted_prompts.append("")
            continue
            
        user_message = f"{system_prompt}\n\nТекст:\n{text}"
        prompt = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
        formatted_prompts.append(prompt)
    return formatted_prompts

def initialize_model(model_id: str = "google/gemma-2-9b-it"):
    """Инициализирует vLLM модель с оптимальными параметрами для Colab/T4."""
    print(f"Инициализация модели {model_id}...")
    llm = LLM(
        model=model_id, 
        tensor_parallel_size=1, 
        dtype="bfloat16",
        gpu_memory_utilization=0.9, 
        max_model_len=4096, 
        #quantization="awq",
    )
    return llm

def generate_responses(llm: LLM, formatted_prompts: list, max_tokens: int = 512) -> list:
    """Запускает батчевую генерацию и возвращает список сгенерированных ответов."""
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=max_tokens
    )
    
    print(f"Начинаем батчевую генерацию для {len(formatted_prompts)} текстов...")
    start_time = time.time()
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    print(f"Генерация завершена за {time.time() - start_time:.2f} секунд")
    
    # Извлечение результатов
    generated_prompts = [output.outputs[0].text.strip() for output in outputs]
    return generated_prompts

def save_results(df: pd.DataFrame, generated_prompts: list, output_file: str):
    """Сохраняет результаты в новый CSV файл."""
    df['promt'] = generated_prompts
    df.to_csv(output_file, index=False, sep=';')
    print(f"Успешно сохранено в {output_file}")

def main():
    # 1. Настройка путей
    input_file = '../drive/MyDrive/Диплом/datasets/processed_posts.csv'
    output_file = '../drive/MyDrive/Диплом/datasets/processed_posts_with_prompts.csv'
    
    # 2. Загрузка
    df = load_data(input_file)
    texts = df['clean_text'].fillna("").tolist()
    
    # 3. Форматирование
    system_prompt = "Ты AI-ассистент. По готовому тексту восстанови короткий и четкий промт (запрос), по которому этот текст мог бы быть сгенерирован. В ответ напиши только сам промт."
    formatted_prompts = format_prompts(texts, system_prompt)
    
    # 4. Инициализация модели
    model_id = "google/gemma-4-E2B-it" 
    llm = initialize_model(model_id)
    
    # 5. Генерация
    generated_prompts = generate_responses(llm, formatted_prompts, max_tokens=512)
    
    # 6. Сохранение
    save_results(df, generated_prompts, output_file)

if __name__ == "__main__":
    main()

