"""
Общая конфигурация для пайплайна дообучения Gemma 4 E2B IT.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    """Пути к данным и моделям."""
    # Данные
    data_dir: str = "Data"
    sft_dataset: str = "Data/sft_dataset.jsonl"
    ipo_dataset: str = "Data/ipo_dataset.jsonl"

    # Модели
    base_model: str = "google/gemma-4-E2B-it"
    sft_output_dir: str = "models/sft-gemma-4-e2b"
    ipo_output_dir: str = "models/ipo-gemma-4-e2b"

    # Google Drive (для Colab)
    drive_base: str = "/content/drive/MyDrive/Диплом"
    drive_datasets: str = "/content/drive/MyDrive/Диплом/datasets"
    drive_models: str = "/content/drive/MyDrive/Диплом/models"


@dataclass
class LoraConfig:
    """Параметры QLoRA адаптеров."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


@dataclass
class SFTHyperParams:
    """Гиперпараметры для SFT-стадии."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class IPOHyperParams:
    """Гиперпараметры для IPO-стадии."""
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 512
    beta: float = 0.1  # IPO regularization
    loss_type: str = "ipo"
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 3
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    seed: int = 42
