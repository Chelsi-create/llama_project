# inf.py
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load model & tokenizer once
model_dir = "lora_llama3_sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# Optional label mapping
id2label = {0: "positive", 1: "negative"}

class InputText(BaseModel):
    text: str

def run_inference(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "label": id2label[pred],
        "confidence": round(confidence * 100, 2)
    }
