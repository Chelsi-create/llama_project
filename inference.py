from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch

# Load tokenizer and base model
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "lora_llama3_sentiment"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base + LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path, config=config)
model.eval()

# Sentiment label mapping
id2label = {1: "negative", 0: "positive"}

# Inference function
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    return id2label[pred]

# --- EXAMPLES ---
if __name__ == "__main__":
    samples = [
  "The product quality is top-notch. Will buy again!",
  "This is the worst experience Ive had shopping online.",
  "Fast shipping and great customer service!",
  "Totally not worth the money. I want a refund.",
  "Incredible performance, very responsive app!",
  "The item arrived broken. Very poor packaging.",
  "Super easy to install and works perfectly.",
  "Terrible design and cheap materials.",
  "It works fine but nothing special.",
  "Got the job done, but I wouldn't recommend it.",
  "This product completely failed after just a week of use.",
    "I had a terrible experience with their customer support.",
    "Not worth the money — it broke on day one.",
    "Cheaply made and doesn't perform as advertised.",
    "Very disappointed. I regret buying this.",
    "The app crashes every time I try to open it.",
    "It looks good but functions horribly.",
    "Delivery was late and the packaging was damaged.",
    "The quality is terrible and the sound is distorted.",
    "Setup was confusing and nothing worked as expected."
]

    for s in samples:
        print(f"[{s}] → {classify(s)}")
