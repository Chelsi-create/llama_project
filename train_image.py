from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from glob import glob

# --- 1. Reproducibility ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# --- 2. Accelerator setup (use bf16 if supported) ---
accelerator = Accelerator(mixed_precision="bf16")  # or "fp16" if not on H100/A100
device = accelerator.device

# --- 3. Load COCO dataset instead of uploaded images ---
print("Loading COCO dataset...")

# Load COCO dataset with a subset of categories for faster training
# You can modify these categories based on what you want to classify
selected_categories = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow'
]

try:
    # Load COCO dataset from Hugging Face
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=False)
    print(f"Loaded COCO dataset with {len(dataset)} images")
    
    # Filter and prepare data
    filtered_data = []
    class_counts = {cat: 0 for cat in selected_categories}
    max_samples_per_class = 200  # Limit samples per class for faster training
    
    print("Filtering COCO dataset...")
    for idx, sample in enumerate(tqdm(dataset)):
        if idx > 10000:  # Limit total samples to process
            break
            
        # Get annotations
        annotations = sample.get('objects', {})
        categories = annotations.get('category', [])
        
        # Check if any of our selected categories are present
        for cat_id, category in enumerate(categories):
            if isinstance(category, int):
                # Convert category ID to name (this might need adjustment based on COCO format)
                continue
            elif isinstance(category, str) and category in selected_categories:
                if class_counts[category] < max_samples_per_class:
                    filtered_data.append({
                        'image': sample['image'],
                        'label': category
                    })
                    class_counts[category] += 1
                    break
    
    print(f"Filtered dataset size: {len(filtered_data)}")
    print("Class distribution:")
    for cat, count in class_counts.items():
        if count > 0:
            print(f"  {cat}: {count} images")
    
    # Create class mapping
    used_classes = [cat for cat, count in class_counts.items() if count > 0]
    if not used_classes:
        raise RuntimeError("No images found for selected categories")
    
    class_names = sorted(used_classes)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Prepare image paths and labels
    image_data = []
    labels = []
    
    for item in filtered_data:
        if item['label'] in class_to_idx:
            image_data.append(item['image'])
            labels.append(class_to_idx[item['label']])
    
except Exception as e:
    print(f"Error loading COCO dataset: {e}")
    print("Falling back to a simple synthetic dataset...")
    
    # Fallback: Create a simple synthetic dataset
    class_names = ['red_square', 'blue_circle', 'green_triangle']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Create synthetic images
    image_data = []
    labels = []
    
    for class_name, class_idx in class_to_idx.items():
        for i in range(100):  # 100 images per class
            # Create a simple colored shape
            img = Image.new('RGB', (224, 224), color='white')
            # Add some simple patterns based on class
            if 'red' in class_name:
                img = Image.new('RGB', (224, 224), color='red')
            elif 'blue' in class_name:
                img = Image.new('RGB', (224, 224), color='blue')
            elif 'green' in class_name:
                img = Image.new('RGB', (224, 224), color='green')
            
            image_data.append(img)
            labels.append(class_idx)

print(f"Final dataset: {len(image_data)} images across {len(class_names)} classes")
print(f"Classes: {class_names}")

# Initialize processor
model_ckpt = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_ckpt)

# Custom Dataset class for COCO/synthetic data
from torch.utils.data import Dataset
class COCOImageDataset(Dataset):
    def __init__(self, image_data, labels, processor):
        self.image_data = image_data
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        try:
            # Handle both PIL images and potentially other formats
            img = self.image_data[idx]
            if not isinstance(img, Image.Image):
                # Convert to PIL if necessary
                img = Image.fromarray(img) if hasattr(img, 'shape') else img
            
            # Ensure RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            processed = self.processor(img, return_tensors="pt")
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "label": self.labels[idx]
            }
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            # Return a dummy image in case of error
            dummy_img = Image.new('RGB', (224, 224), color='black')
            processed = self.processor(dummy_img, return_tensors="pt")
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "label": self.labels[idx]
            }

# Shuffle and split dataset
indices = list(range(len(image_data)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

train_dataset = COCOImageDataset(
    [image_data[i] for i in train_idx], 
    [labels[i] for i in train_idx], 
    processor
)
test_dataset = COCOImageDataset(
    [image_data[i] for i in test_idx], 
    [labels[i] for i in test_idx], 
    processor
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# --- 5. DataLoaders ---
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)  # Reduced batch size
eval_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# --- 6. Load ViT model ---
model = ViTForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# --- 7. Unfreeze only last 2 ViT layers ---
for name, param in model.vit.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "layernorm" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Ensure classifier head is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Print number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# --- 8. Optimizer ---
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)

# --- 9. Prepare everything with accelerator ---
model, optimizer, train_loader, eval_loader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

# --- 10. Training loop ---
num_epochs = 1
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=not accelerator.is_local_main_process):
        try:
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    accelerator.print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0
    num_eval_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            try:
                outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                eval_loss += outputs.loss.item()
                num_eval_batches += 1
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    accuracy = 100 * correct / max(total, 1)
    avg_eval_loss = eval_loss / max(num_eval_batches, 1)
    accelerator.print(f"Epoch {epoch+1} | Accuracy: {accuracy:.2f}% | Eval Loss: {avg_eval_loss:.4f}")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        accelerator.print(f"New best accuracy: {best_accuracy:.2f}%")

# --- 11. Save model & processor ---
if accelerator.is_main_process:
    os.makedirs("../outputs/image_checkpoints", exist_ok=True)
    
    # Unwrap model if using accelerator
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("../outputs/image_checkpoints")
    processor.save_pretrained("../outputs/image_checkpoints")
    
    # Save class names for inference
    import json
    with open("../outputs/image_checkpoints/class_names.json", "w") as f:
        json.dump(class_names, f)
    
    # Save training info
    training_info = {
        "classes": class_names,
        "num_classes": len(class_names),
        "best_accuracy": best_accuracy,
        "total_samples": len(image_data),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "epochs": num_epochs,
        "model_checkpoint": model_ckpt
    }
    
    with open("../outputs/image_checkpoints/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    accelerator.print("✅ Model and processor saved to ../outputs/image_checkpoints")
    accelerator.print(f"✅ Final accuracy: {best_accuracy:.2f}%")
    accelerator.print(f"✅ Classes trained: {class_names}")