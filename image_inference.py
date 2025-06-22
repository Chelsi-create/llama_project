from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import base64
import io
import json
import os
import numpy as np

class WebcamImageClassifier:
    def __init__(self, model_path="../outputs/image_checkpoints"):
        """
        Initialize the webcam image classifier
        
        Args:
            model_path: Path to the saved model and processor
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model, processor, and class names"""
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                print(f"Model path {self.model_path} does not exist")
                return False
            
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"No model config found at {config_path}")
                return False
            
            # Load class names
            class_names_path = os.path.join(self.model_path, "class_names.json")
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
            else:
                print("No class names file found")
                return False
            
            # Load processor
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            print("✅ Processor loaded successfully")
            
            # Load model
            self.model = ViTForImageClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.class_names)
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.processor = None
            self.class_names = []
            return False
    
    def predict_base64_image(self, base64_image):
        """
        Predict class for a base64 encoded image
        
        Args:
            base64_image: Base64 encoded image string (with or without data:image prefix)
            
        Returns:
            dict: Prediction results with label, confidence, and probabilities
        """
        try:
            if self.model is None or self.processor is None:
                return {"error": "Model not loaded"}
            
            # Decode base64 image
            if base64_image.startswith('data:image'):
                # Remove data URL prefix
                base64_image = base64_image.split(',')[1]
            
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess image
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get results
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item() * 100
            
            if predicted_idx < len(self.class_names):
                predicted_label = self.class_names[predicted_idx]
            else:
                predicted_label = f"Unknown_Class_{predicted_idx}"
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(3, len(self.class_names)))
            top_predictions = []
            
            for prob, idx in zip(top_probs, top_indices):
                if idx.item() < len(self.class_names):
                    label = self.class_names[idx.item()]
                else:
                    label = f"Unknown_Class_{idx.item()}"
                
                top_predictions.append({
                    "label": label,
                    "confidence": prob.item() * 100
                })
            
            return {
                "label": predicted_label,
                "confidence": confidence_score,
                "top_predictions": top_predictions,
                "class_probabilities": {
                    self.class_names[i]: probabilities[0][i].item() * 100 
                    for i in range(min(len(self.class_names), len(probabilities[0])))
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_pil_image(self, pil_image):
        """
        Predict class for a PIL Image
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            dict: Prediction results
        """
        try:
            if self.model is None or self.processor is None:
                return {"error": "Model not loaded"}
            
            # Ensure RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess image
            inputs = self.processor(pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get results
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item() * 100
            
            if predicted_idx < len(self.class_names):
                predicted_label = self.class_names[predicted_idx]
            else:
                predicted_label = f"Unknown_Class_{predicted_idx}"
            
            return {
                "label": predicted_label,
                "confidence": confidence_score,
                "predicted_idx": predicted_idx
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_loaded": True,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "device": str(self.device),
            "model_path": self.model_path
        }

# Legacy functions for backward compatibility
def classify_image(base64_image):
    """
    Legacy function for image classification
    
    Args:
        base64_image: Base64 encoded image
        
    Returns:
        str: Classification result
    """
    classifier = WebcamImageClassifier()
    if classifier.model is None:
        return "No model available"
    
    result = classifier.predict_base64_image(base64_image)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return f"{result['label']} ({result['confidence']:.1f}%)"

def few_shot_classify_image(base64_image, examples):
    """
    Legacy function for few-shot image classification
    Note: This is a placeholder implementation
    
    Args:
        base64_image: Base64 encoded image
        examples: List of example images and labels
        
    Returns:
        str: Classification result
    """
    # For now, just use the regular classification
    # In a full implementation, you would use the examples for few-shot learning
    return classify_image(base64_image)

# Test function
def test_classifier():
    """Test the classifier with a dummy image"""
    classifier = WebcamImageClassifier()
    
    if classifier.model is None:
        print("❌ No model loaded for testing")
        return False
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64 for testing
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode()
    
    # Test prediction
    result = classifier.predict_base64_image(base64_image)
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False
    
    print(f"✅ Test successful: {result['label']} ({result['confidence']:.1f}%)")
    return True

if __name__ == "__main__":
    # Test the classifier when run directly
    print("Testing WebcamImageClassifier...")
    test_classifier()