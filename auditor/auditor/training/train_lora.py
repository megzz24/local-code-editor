import json
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig  # type: ignore[reportPrivateImportUsage]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024    
TRAIN_FILE = PROJECT_ROOT / "dataset" / "train.jsonl"
LORA_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"
MERGED_MODEL_DIR = PROJECT_ROOT / "outputs" / "merged_model"

LORA_R = 16                    
LORA_ALPHA = 16
LORA_DROPOUT = 0.0              
SAVE_MERGED_MODEL = False

NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4

def load_base_model():
    print(f"Loading base model: {BASE_MODEL_NAME} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,        
        load_in_4bit=True,   
    )
    return model, tokenizer

def add_lora_adapters(model):
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves VRAM
        random_state=42,
    )
    return model

def load_dataset(tokenizer):
    print(f"Loading training data from {TRAIN_FILE}...")

    raw_examples = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_examples.append(json.loads(line))

    print(f"Loaded {len(raw_examples)} examples.")
    texts = []
    for ex in raw_examples:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})

def train(model, tokenizer, dataset):
    print("Starting training...")
    config_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir=str(PROJECT_ROOT / "outputs" / "checkpoints"),
        save_strategy="no",  
        report_to="none",
        seed=42,
    )
    config = SFTConfig(**config_kwargs) 

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  
        train_dataset=dataset,
        args=config,
    )

    trainer.train()
    return trainer


def is_bf16_supported():
    try:
        from unsloth import is_bfloat16_supported
        return is_bfloat16_supported()
    except Exception:
        return False

def save_model(model, tokenizer):
    LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter to {LORA_ADAPTER_DIR}...")
    model.save_pretrained(str(LORA_ADAPTER_DIR))
    tokenizer.save_pretrained(str(LORA_ADAPTER_DIR))

    if not SAVE_MERGED_MODEL:
        print(
            "Skipping merged model save (SAVE_MERGED_MODEL=False). "
            f"The LoRA adapter in {LORA_ADAPTER_DIR} is sufficient for "
            "inference - see services/auditor/inference.py."
        )
        return

    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {MERGED_MODEL_DIR}...")
    model.save_pretrained_merged(
        str(MERGED_MODEL_DIR),
        tokenizer,
        save_method="merged_16bit",
    )

def main():
    model, tokenizer = load_base_model()
    model = add_lora_adapters(model)
    dataset = load_dataset(tokenizer)
    train(model, tokenizer, dataset)
    save_model(model, tokenizer)
    print("Done! LoRA adapter and merged model saved under outputs/.")


if __name__ == "__main__":
    main()