
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from typing import Dict

# Import our engine
from model_engine import process_image_data, load_model, MODEL_PATH

app = FastAPI(title="SmartPlant Vision API", version="2.1")

# Allow CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Preload model on startup to speed up first request
    if os.path.exists(MODEL_PATH):
        load_model(MODEL_PATH)
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found!")

@app.get("/")
def read_root():
    return {"status": "online", "service": "SmartPlant Vision API"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = process_image_data(contents)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
