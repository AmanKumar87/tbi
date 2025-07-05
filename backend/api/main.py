import io
import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.ml.models.hybrid_cnn import create_hybrid_cnn
from backend.ml.models.vision_transformer import create_vision_transformer
from backend.ml.models.high_accuracy_hybrid import create_high_accuracy_hybrid
from backend.ml.utils.data_loader import val_test_transforms

# --- App Setup ---
app = FastAPI(title="TBI Prediction API")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- All your existing backend logic for model loading and API endpoints ---
# (The code for /predict, /summarise, TBI_INFO, etc. goes here)
# ...
# ...
# --- End of existing logic ---


# --- FINAL STEP: SERVE THE FRONTEND ---

# Define the path to the frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# Mount the 'static' folder (for CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

# This catch-all route serves all your HTML files
@app.get("/{file_path:path}")
async def serve_frontend(file_path: str = "index.html"):
    path = os.path.join(FRONTEND_DIR, file_path)
    if not os.path.isfile(path):
        # If the path is not a file, default to index.html
        return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))
    return FileResponse(path)

# Ensure the root path also serves index.html
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))