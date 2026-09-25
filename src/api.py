import os
import json
import tempfile
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING, get_device
from src.infer import run_inference_on_file, write_raven_selections

# Initialize FastAPI App
app = FastAPI(
    title="Baleen Whale Sound Event Detection API",
    description="REST API for classifying and localizing Southern Ocean baleen whale vocalizations in audio files.",
    version="1.0.0"
)

# Global variables for model state
model = None
device = None
default_thresholds = {}

class DetectionEvent(BaseModel):
    cls_name: str
    begin_time_sec: float
    end_time_sec: float
    begin_samp: int
    end_samp: int
    confidence: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

@app.on_event("startup")
def startup_event():
    global model, device, default_thresholds
    device = get_device()
    model_path = "models/best_model.pth"
    
    # Initialize WhaleSEDModel
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        print(f"Model successfully loaded on {device} from {model_path}")
        
        # Load thresholds JSON
        thresholds_path = model_path.replace(".pth", "_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                default_thresholds = json.load(f)
            print(f"Loaded optimal thresholds: {default_thresholds}")
        else:
            default_thresholds = {cls: 0.5 for cls in CLASS_MAPPING.keys()}
    else:
        print(f"Warning: No model checkpoint found at {model_path}. Endpoints will fail until trained.")
        default_thresholds = {cls: 0.5 for cls in CLASS_MAPPING.keys()}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns service health status and computational device properties."""
    return HealthResponse(
        status="healthy",
        device=str(device) if device else "none",
        model_loaded=(model is not None and next(model.parameters()).is_cuda or torch.backends.mps.is_available() or device is not None)
    )

def execute_inference(file: UploadFile, thresholds_override: Optional[str] = None):
    """Saves file to temporary scratch location, runs inference, and returns raw events."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model first.")
        
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only standard .wav audio files are supported.")
        
    # Unpack custom thresholds if passed as a serialized JSON string
    thresholds = default_thresholds.copy()
    if thresholds_override:
        try:
            custom_th = json.loads(thresholds_override)
            for k, v in custom_th.items():
                if k in thresholds:
                    thresholds[k] = float(v)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid thresholds_override JSON format: {e}")
            
    # Save uploaded file to temp file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run sliding window inference
        events = run_inference_on_file(
            model=model,
            wav_path=temp_file_path,
            device=device,
            thresholds=thresholds
        )
        return events, temp_file_path, temp_dir
    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict")
def predict_events(
    file: UploadFile = File(...),
    thresholds_override: Optional[str] = Query(None, description="JSON string mapping classes to thresholds, e.g. '{\"Bm.Ant-A\":0.15}'")
):
    """
    Accepts a WAV file upload, runs sound event detection,
    and returns a JSON list of localized event objects.
    """
    events, temp_file, temp_dir = execute_inference(file, thresholds_override)
    
    # Cleanup temp resources
    shutil.rmtree(temp_dir)
    
    response_events = []
    for ev in events:
        response_events.append({
            "cls_name": ev["class"],
            "begin_time_sec": ev["begin_time_sec"],
            "end_time_sec": ev["end_time_sec"],
            "begin_samp": ev["begin_samp"],
            "end_samp": ev["end_samp"],
            "confidence": ev["confidence"]
        })
        
    return JSONResponse(content={"filename": file.filename, "detections": response_events})

@app.post("/predict/selections")
def predict_selections_table(
    file: UploadFile = File(...),
    thresholds_override: Optional[str] = Query(None, description="JSON string mapping classes to thresholds")
):
    """
    Accepts a WAV file upload, runs sound event detection,
    and returns a downloadable Raven-compatible selection table (.txt).
    """
    events, temp_file, temp_dir = execute_inference(file, thresholds_override)
    
    # Generate selections table in temp directory
    out_name = file.filename.replace(".wav", ".predictions.selections.txt")
    out_path = os.path.join(temp_dir, out_name)
    
    write_raven_selections(events, out_path, file.filename)
    
    # We return the FileResponse, but need to clean up the temp directory AFTER response is sent.
    # FastAPI's FileResponse handles reading, so we can clean up in a background task,
    # or let the temp file clean up happen manually. To prevent leaks, we can write a custom response
    # wrapper or background task. FastAPI BackgroundTasks is perfect here.
    from fastapi import BackgroundTasks
    
    def cleanup_temp():
        shutil.rmtree(temp_dir)
        
    background_tasks = BackgroundTasks()
    background_tasks.add_task(cleanup_temp)
    
    return FileResponse(
        path=out_path,
        media_type="text/plain",
        filename=out_name,
        background=background_tasks
    )
