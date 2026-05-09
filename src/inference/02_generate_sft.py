import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from pathlib import Path

def main():
    # Настройки путей
    TEST_DATA_PATH = '../drive/MyDrive/diplom/datasets/processed_posts_with_prompts_test.csv'
    OUTPUT_PATH = '../drive/MyDrive/diplom/datasets/generated_sft.csv'
    
    # Укажите путь к базовой модели и путь к сохраненному LoRA адаптеру SFT
    BASE_MODEL_PATH = "google/gemma-4-E2B-it" # Или другая ваша базовая модель
    LORA_PATH = "../drive/MyDrive/diplom/diplom/Models/gemma-4-E2B-it-SFT" # ПУТЬ К ПАПКЕ С adapter_config.json

    print(f"Загрузка тестовых данных из {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, sep=";")
    
    # df = df.head(1000)

    prompts = df['promt'].fillna("").tolist()
    
    print(f"Инициализация токенизатора для {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    print("Применение chat_template...")
    formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    print(f"Инициализация vLLM с базовой моделью {BASE_MODEL_PATH} и поддержкой LoRA...")
    llm = LLM(
        model=BASE_MODEL_PATH, 
        max_model_len=4096*2, 
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=64 # Укажите rank, с которым обучали LoRA (если больше 64, увеличьте)
    )
    
    sampling_params = SamplingParams(
        temperature=1, 
        top_p=0.95, 
        max_tokens=1024,
        repetition_penalty=1.1
    )

    print(f"Начало генерации с применением адаптера из {LORA_PATH}...")
    # Применяем LoRA адаптер "на лету" во время генерации
    outputs = llm.generate(
        formatted_prompts, 
        sampling_params,
        lora_request=LoRARequest("sft_adapter", 1, LORA_PATH)
    )
    
    generated_texts = [out.outputs[0].text.strip() for out in outputs]
    df['generated_sft'] = generated_texts
    
    df.to_csv(OUTPUT_PATH, sep=";", index=False)
    print(f"Готово! Результаты сохранены в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
