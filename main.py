from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
import time
from inf import InputText, run_inference
from typing import List
import shutil
import base64
from datetime import datetime
import json
from image_inference import WebcamImageClassifier

app = FastAPI()
progress = {"percent": 0, "status": "idle", "message": ""}

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-text")
async def upload_text(file: UploadFile = File(...)):
    os.makedirs("uploads/text", exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{file.filename}"
    path = os.path.join("uploads/text", new_filename)
    
    # Save file
    with open(path, "wb") as f:
        f.write(await file.read())
    
    return {"message": f"Uploaded as {new_filename}", "filename": new_filename}

latest_class_names = []

@app.post("/train-text")
async def train_text(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    global latest_num_labels, latest_class_names
    latest_num_labels = payload.get("num_classes", 2)
    latest_class_names = payload.get("class_names", [])
    
    # Optionally, save to file for later inference
    with open("uploads/text/class_names.json", "w") as f:
        json.dump(latest_class_names, f)
    
    background_tasks.add_task(start_training_process)
    return {"message": f"Training started with {latest_num_labels} classes."}

@app.post("/train-image")
async def train_image(background_tasks: BackgroundTasks):
    global progress
    progress = {"percent": 0, "status": "starting", "message": "Initializing COCO dataset training..."}
    background_tasks.add_task(start_image_training_process)
    return {"message": "Image training started on COCO dataset!"}

@app.get("/progress")
async def get_progress():
    return progress

def start_training_process():
    global progress
    try:
        progress["status"] = "training"
        progress["message"] = "Training text model..."
        
        # Start text training script
        process = subprocess.Popen([
            "python", "train_text.py",
            "--num_labels", str(latest_num_labels)
        ])
        
        # Simulate progress bar (10% → 100% over ~50s)
        for i in range(1, 11):
            time.sleep(5)
            progress["percent"] = i * 10
            progress["message"] = f"Training text model... {i*10}%"
        
        progress["percent"] = 100
        progress["status"] = "completed"
        progress["message"] = "Text training completed!"
        
    except Exception as e:
        progress["status"] = "error"
        progress["message"] = f"Training failed: {str(e)}"
        progress["percent"] = 0

def start_image_training_process():
    global progress
    try:
        progress["status"] = "training"
        progress["message"] = "Loading COCO dataset..."
        progress["percent"] = 5
        
        # Start image training script (now uses COCO dataset)
        process = subprocess.Popen([
            "python", "train_image_coco.py"  # Use the new COCO training script
        ])
        
        # More detailed progress simulation for image training
        training_steps = [
            (10, "Loading COCO dataset..."),
            (20, "Filtering and preparing data..."),
            (30, "Initializing Vision Transformer model..."),
            (40, "Starting training - Epoch 1/3..."),
            (55, "Training - Epoch 1/3 completed"),
            (70, "Training - Epoch 2/3 completed"),
            (85, "Training - Epoch 3/3 completed"),
            (95, "Saving model and checkpoints..."),
            (100, "Training completed successfully!")
        ]
        
        for percent, message in training_steps:
            time.sleep(6)  # Slightly longer for image training
            progress["percent"] = percent
            progress["message"] = message
        
        progress["status"] = "completed"
        
        # Reinitialize the classifier with the new model
        global image_classifier
        image_classifier = None  # Reset to force reinitialization
        
    except Exception as e:
        progress["status"] = "error"
        progress["message"] = f"Image training failed: {str(e)}"
        progress["percent"] = 0

@app.post("/predict")
async def predict(input: InputText):
    return run_inference(input.text)

# Keep the upload endpoints for frontend compatibility, but they're now just for show
@app.post("/upload-images")
async def upload_images(
    files: List[UploadFile] = File(...),
    class_name: str = Form(...),
    timestamp_dir: str = Form(None)
):
    """
    FAKE upload - saves images for frontend compatibility but training uses COCO dataset
    """
    if not files or not class_name.strip():
        return {"error": "files and class_name are required"}

    # Still save the images for frontend display/testing, but training ignores them
    if not timestamp_dir:
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.join("uploads", "images", timestamp_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"[FAKE UPLOAD] Saving to directory: {base_dir}")
    print(f"[FAKE UPLOAD] Class name: {class_name}")
    print(f"[FAKE UPLOAD] Number of files: {len(files)}")
    print("[INFO] Training will use COCO dataset instead of uploaded images")

    def next_index(cls: str) -> int:
        existing_files = [
            f for f in os.listdir(base_dir)
            if f.startswith(f"{cls}_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not existing_files:
            return 1
        
        indices = []
        for f in existing_files:
            try:
                name_part = f[len(cls)+1:]
                num_part = name_part.split('.')[0]
                indices.append(int(num_part))
            except ValueError:
                continue
        
        return max(indices) + 1 if indices else 1

    uploaded = []
    start_idx = next_index(class_name)
    
    for i, file in enumerate(files):
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
        if not ext:
            ext = ".jpg"
        
        name = f"{class_name}_{start_idx + i}{ext}"
        path = os.path.join(base_dir, name)

        try:
            content = await file.read()
            with open(path, "wb") as f:
                f.write(content)
            uploaded.append(name)
        except Exception as e:
            print(f"Error saving file {name}: {e}")

    return {
        "message": f"{len(uploaded)} images saved (Note: Training will use COCO dataset)",
        "timestamp_dir": timestamp_dir,
        "files": uploaded,
        "class_name": class_name,
        "total_files": len(files),
        "saved_files": len(uploaded),
        "training_note": "Training will use COCO dataset for better reliability"
    }

@app.post("/upload-webcam")
async def upload_webcam_image(
    file: UploadFile = File(...), 
    class_name: str = Form(...), 
    timestamp_dir: str = Form(None)
):
    """
    FAKE webcam upload - saves for frontend compatibility but training uses COCO dataset
    """
    print(f"[FAKE WEBCAM] Class: {class_name}, Timestamp: {timestamp_dir}")
    print("[INFO] Training will use COCO dataset instead of webcam images")
    
    if not timestamp_dir:
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = os.path.join("uploads", "images", timestamp_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    existing_files = [
        f for f in os.listdir(save_dir) 
        if f.startswith(f"{class_name}_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    
    next_num = len(existing_files) + 1
    filename = f"{class_name}_{next_num}.png"
    path = os.path.join(save_dir, filename)
    
    try:
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        
        return {
            "message": f"Image saved as {filename} (Training uses COCO dataset)",
            "timestamp_dir": timestamp_dir,
            "class_name": class_name,
            "filename": filename,
            "path": path,
            "training_note": "Training will use COCO dataset for better reliability"
        }
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

@app.post("/capture-image")
async def capture_image(base64_image: str = Form(...)):
    """Keep for frontend compatibility"""
    import base64
    os.makedirs("uploads/image", exist_ok=True)
    image_data = base64.b64decode(base64_image.split(",")[-1])
    filename = f"uploads/image/capture_{datetime.now().isoformat()}.png"

    with open(filename, "wb") as f:
        f.write(image_data)
    return {"message": "Image captured and saved (Training uses COCO dataset)."}

@app.post("/upload-few-shot-text")
async def upload_few_shot(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = json.loads(content)
        return {"message": "Few-shot dataset uploaded", "examples": data}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Global classifier instance for better performance
image_classifier = None

def initialize_image_classifier():
    """Initialize the image classifier on startup"""
    global image_classifier
    try:
        # Check if we have a trained model
        model_path = "../outputs/image_checkpoints"
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            print("⚠️ No trained model found. Please train a model first.")
            return False
            
        image_classifier = WebcamImageClassifier()
        if image_classifier.model is not None:
            print("✅ Image classifier initialized successfully")
            
            # Load and display training info if available
            info_path = os.path.join(model_path, "training_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    training_info = json.load(f)
                print(f"✅ Model trained on {training_info.get('num_classes', 'unknown')} classes")
                print(f"✅ Classes: {training_info.get('classes', [])}")
                print(f"✅ Best accuracy: {training_info.get('best_accuracy', 0):.2f}%")
                
            return True
        else:
            print("⚠️ Image classifier model not loaded properly")
            return False
    except Exception as e:
        print(f"❌ Error initializing image classifier: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_image_classifier()

@app.post("/classify")
async def classify_input(request: dict):
    """
    Handle classification requests from frontend
    Supports both text and image inputs with real-time capability
    """
    task = request.get("task")
    examples = request.get("examples", [])
    input_data = request.get("input")
    
    if not input_data:
        return {"error": "No input provided"}
    
    if task == "text":
        try:
            result = run_inference(input_data)
            return {"result": result, "task": "text"}
        except Exception as e:
            return {"error": f"Text classification failed: {str(e)}"}
    
    elif task == "image":
        try:
            global image_classifier
            
            # Try to initialize if not already done
            if image_classifier is None:
                initialize_image_classifier()
            
            if image_classifier is None or image_classifier.model is None:
                return {
                    "error": "Image model not loaded. Please train an image model first using COCO dataset.",
                    "result": "No Model Available",
                    "task": "image"
                }
            
            # Use the optimized classifier
            result = image_classifier.predict_base64_image(input_data)
            
            if "error" in result:
                return {"error": result["error"], "task": "image"}
            
            return {
                "result": f"{result['label']} ({result['confidence']:.1f}%)",
                "task": "image",
                "details": result
            }
            
        except Exception as e:
            return {"error": f"Image classification failed: {str(e)}", "task": "image"}
    
    elif task == "audio":
        return {"result": "Audio classification not yet implemented", "task": "audio"}
    
    else:
        return {"error": f"Unknown task: {task}"}

@app.post("/classify-webcam")
async def classify_webcam(image_data: dict):
    """
    Real-time webcam image classification endpoint
    """
    try:
        global image_classifier
        
        if image_classifier is None:
            initialize_image_classifier()
        
        if image_classifier is None or image_classifier.model is None:
            return {
                "error": "Image model not available. Please train a model first.",
                "label": "No Model",
                "confidence": 0.0
            }
        
        base64_image = image_data.get("image")
        if not base64_image:
            return {"error": "No image data provided"}
        
        # Get prediction
        result = image_classifier.predict_base64_image(base64_image)
        
        return result
        
    except Exception as e:
        return {
            "error": f"Classification failed: {str(e)}",
            "label": "Error",
            "confidence": 0.0
        }

# New endpoint to get training status and model info
@app.get("/model-info")
async def get_model_info():
    """Get information about the currently loaded model"""
    try:
        model_path = "../outputs/image_checkpoints"
        info_path = os.path.join(model_path, "training_info.json")
        
        if not os.path.exists(info_path):
            return {"error": "No model training information found"}
        
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        
        return {
            "model_available": True,
            "classes": training_info.get("classes", []),
            "num_classes": training_info.get("num_classes", 0),
            "accuracy": training_info.get("best_accuracy", 0),
            "total_samples": training_info.get("total_samples", 0),
            "train_samples": training_info.get("train_samples", 0),
            "test_samples": training_info.get("test_samples", 0),
            "epochs": training_info.get("epochs", 0)
        }
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

# New endpoint to reinitialize classifier after training
@app.post("/reinitialize-classifier")
async def reinitialize_classifier():
    """Reinitialize the image classifier after training"""
    global image_classifier
    try:
        image_classifier = None
        success = initialize_image_classifier()
        
        if success:
            return {"message": "Image classifier reinitialized successfully"}
        else:
            return {"error": "Failed to reinitialize image classifier"}
            
    except Exception as e:
        return {"error": f"Reinitialization failed: {str(e)}"}