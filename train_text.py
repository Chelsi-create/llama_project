from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
import json
import os

upload_dir = "uploads/text"

# Get all files in the directory
all_files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir)]

# Filter for .json files that are not class_names.json and are actual files
json_files = [
    f for f in all_files 
    if f.endswith(".json") and "class_names.json" not in f and os.path.isfile(f)
]

if not json_files:
    print("No suitable .json files found in uploads/text.")
    exit()

# Find the latest file based on modification time
latest_file = max(json_files, key=os.path.getmtime)

print(f"Using latest file: {latest_file}")

# Load dataset
with open(latest_file, "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# --- 2. Load tokenizer ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# --- 3. Format & tokenize ---
def tokenize(example):
    # Format for instruction-tuned model
    formatted = f"[INST] Classify the sentiment of the following: {example['text']} [/INST]"
    encoding = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    encoding["labels"] = example["label"]
    return encoding

tokenized = dataset.map(tokenize, remove_columns=["text", "label"])

# --- 4. LoRA config ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.SEQ_CLS,
    lora_dropout=0.1,
    bias="none",
    use_rslora=False,
    init_lora_weights=True
)

# --- 5. Load base model + apply LoRA ---
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = get_peft_model(base_model, peft_config)

# --- 6. Training arguments ---
training_args = TrainingArguments(
    output_dir="./lora_llama3_sentiment",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    warmup_ratio=0.1,
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False
)

# --- 7. Trainer ---
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized
)

# --- 8. Train ---
trainer.train()

# --- 9. Save final model ---
print("Saving model...")
model.save_pretrained("./lora_llama3_sentiment")
tokenizer.save_pretrained("./lora_llama3_sentiment")
