import pandas as pd
from vllm import LLM, SamplingParams
import time
import os
from transformers import AutoTokenizer

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


def format_prompts(texts: list, system_prompt: str, model_id: str) -> list:
    """Форматирует тексты в промты для модели (формат Gemma)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    formatted_prompts = []
    for text in texts:
        if not str(text).strip():
            formatted_prompts.append("")
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Текст:\n{text}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    )
    return llm


def generate_responses(llm: LLM, formatted_prompts: list, max_tokens: int = 1024) -> list:
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
    model_id = "google/gemma-4-E2B-it" 
    
    # 2. Загрузка
    df = load_data(input_file)
    texts = df['clean_text'].fillna("").tolist()
    
    # 3. Форматирование
    system_prompt = """Ты — профессиональный редактор. Твоя задача — прочитать готовый пост и написать четкое, лаконичное техническое задание (запрос к ИИ), по которому этот пост был создан.
ПРАВИЛА:
1. Запрос должен быть написан естественным профессиональным языком в виде связного абзаца (1-3 предложения).
2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать списки, буллиты или markdown.
3. Включи в запрос ключевые факты и цифры из текста, чтобы ИИ понимал, о чем конкретно писать.
4. Не пиши инструкции по стилю (не проси "добавить эмодзи" или "написать цепляюще").
ФОРМАТ ОТВЕТА:
Выведи только текст самого запроса, без кавычек и вводных слов.
Начинай с глагола: "Напиши...", "Подготовь...", "Составь...
"""

    formatted_prompts = format_prompts(texts, system_prompt, model_id)
    
    # 4. Инициализация модели
    llm = initialize_model(model_id)
    
    # 5. Генерация
    generated_prompts = generate_responses(llm, formatted_prompts, max_tokens=1024)
    
    # 6. Сохранение
    save_results(df, generated_prompts, output_file)

if __name__ == "__main__":
    main()

