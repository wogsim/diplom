import json
import os

def create_notebook():
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })
        
    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    add_md("# Fine-Tuning Gemma 4 E2B using Unsloth (SFT + IPO)\nThis notebook is optimized for Google Colab Pro.")
    
    add_code("""# 1. Setup Environment
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes""")

    add_code("""# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
# --- ВАЖНО: ИЗМЕНИТЕ ЭТИ ПУТИ ПОД ВАШУ СТРУКТУРУ В GOOGLE DRIVE ---
DRIVE_DIR = '/content/drive/MyDrive/diplom'
SFT_DATA_PATH = os.path.join(DRIVE_DIR, 'Data', 'sft_dataset.jsonl')
IPO_DATA_PATH = os.path.join(DRIVE_DIR, 'Data', 'ipo_dataset.jsonl')
SFT_SAVE_DIR = os.path.join(DRIVE_DIR, 'Models', 'gemma-4-e2b-sft')
IPO_SAVE_DIR = os.path.join(DRIVE_DIR, 'Models', 'gemma-4-e2b-ipo')

os.makedirs(os.path.join(DRIVE_DIR, 'Models'), exist_ok=True)
print("Папки созданы. Можно приступать к обучению.")""")

    add_md("## Stage 1: Supervised Fine-Tuning (SFT)")
    
    add_code("""# 3. Load Model for SFT
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Укажите точное название вашей модели на HuggingFace Hub (например "google/gemma-2-2b-it")
MODEL_NAME = "gemma-4-e2b" 
# Внимание: если модель приватная, нужно будет передать token="Ваш_HF_токен" в from_pretrained

max_seq_length = 2048 # Максимальная длина контекста
dtype = None # Автоматический выбор (bfloat16 для Ampere/Hopper GPU)
load_in_4bit = True # 4-bit квантование для экономии памяти

print("Загрузка базовой модели...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("Настройка PEFT (LoRA)...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Ранг матрицы, 16 - хороший баланс 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # Экономит VRAM при длинном контексте
    random_state = 3407,
)

# Подготовка датасета SFT
print("Загрузка и подготовка SFT датасета...")
sft_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
)

def format_sft(example):
    # Объединяем сообщения пользователя (prompt) и ассистента (completion)
    messages = example["prompt"] + example["completion"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

sft_dataset = sft_dataset.map(format_sft)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

print("Инициализация SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Можно включить True для ускорения если короткие тексты
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Увеличьте max_steps или используйте num_train_epochs = 1 для полного цикла
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_sft",
    ),
)""")

    add_code("""# 4. Запуск SFT-обучения и сохранение в Google Drive
trainer_stats = trainer.train()

model.save_pretrained(SFT_SAVE_DIR)
tokenizer.save_pretrained(SFT_SAVE_DIR)
print(f"SFT адаптеры успешно сохранены в: {SFT_SAVE_DIR}")""")

    add_code("""# 5. Очистка видеопамяти (VRAM) перед IPO-стадией
import gc
del model, tokenizer, trainer, sft_dataset
torch.cuda.empty_cache()
gc.collect()
print("Видеопамять очищена.")""")

    add_md("## Stage 2: Identity Preference Optimization (IPO)")
    
    add_code("""# 6. Загрузка модели с SFT-адаптером
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

print("Загрузка базовой модели для IPO...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, # Загружаем исходную модель
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(f"Применение SFT-адаптера из {SFT_SAVE_DIR}...")
model.load_adapter(SFT_SAVE_DIR)

print("Настройка PEFT (LoRA) для IPO-стадии...")
# Настраиваем обучаемый адаптер поверх SFT
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
)""")

    add_code("""# 7. Подготовка IPO (Preference) датасета
print("Загрузка IPO датасета...")
ipo_dataset = load_dataset("json", data_files=IPO_DATA_PATH, split="train")

def format_ipo(example):
    # Преобразуем prompt в текстовый формат (включая токен генерации)
    prompt_msgs = example["prompt"]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    # Для DPOTrainer (IPO) нужны чистые строки для выбранного (chosen) и отвергнутого (rejected) ответов
    chosen_msgs = example["chosen"]
    rejected_msgs = example["rejected"]
    
    chosen_text = chosen_msgs[0]["content"] if isinstance(chosen_msgs, list) else chosen_msgs
    rejected_text = rejected_msgs[0]["content"] if isinstance(rejected_msgs, list) else rejected_msgs
    
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

ipo_dataset = ipo_dataset.map(format_ipo)

# Проверка формата
print("Пример IPO промпта:", ipo_dataset[0]['prompt'][:100], "...")""")

    add_code("""# 8. Настройка DPOTrainer с loss_type="ipo"
from trl import DPOTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

print("Инициализация DPOTrainer (loss_type='ipo')...")
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None, # В Unsloth не нужно явно передавать ref_model для экономии памяти
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        max_steps = 100, # Увеличьте или используйте num_train_epochs = 1
        learning_rate = 5e-6, # Относительно низкий learning rate для Preference Optimization
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs_ipo",
    ),
    beta = 0.1, # Коэффициент штрафа KL-дивергенции
    loss_type = "ipo", # << ВАЖНО: Используем IPO loss
    train_dataset = ipo_dataset,
    tokenizer = tokenizer,
    max_length = max_seq_length,
    max_prompt_length = max_seq_length // 2,
)""")

    add_code("""# 9. Запуск IPO-обучения и сохранение в Google Drive
dpo_trainer.train()

model.save_pretrained(IPO_SAVE_DIR)
tokenizer.save_pretrained(IPO_SAVE_DIR)
print(f"Конечная IPO модель успешно сохранена в: {IPO_SAVE_DIR}")""")

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/gemma_sft_ipo_colab.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("Создан notebooks/gemma_sft_ipo_colab.ipynb")

if __name__ == "__main__":
    create_notebook()
