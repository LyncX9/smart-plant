
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import random
from datetime import datetime
from typing import List, Dict

# Import our engine
from model_engine import process_image_data, load_model, MODEL_PATH

app = FastAPI(title="SmartPlant Vision API", version="2.2")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if os.path.exists(MODEL_PATH):
        load_model(MODEL_PATH)
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found!")

@app.get("/")
def read_root():
    return {"status": "online", "service": "SmartPlant Vision API"}

@app.post("/analyze")
async def analyze_image(
    images: List[UploadFile] = File(alias="images[]"),
    plant_type: str = Form("rice")
):
    try:
        leaf_results = []
        
        # Aggregation variables
        total_health_score = 0
        total_confidence = 0
        total_lesion_count = 0
        total_lesion_area = 0
        all_class_probs = {}
        
        # Determine number of images
        num_images = len(images)
        if num_images == 0:
             raise HTTPException(status_code=400, detail="No images provided")

        # Process each image
        for idx, file in enumerate(images):
            # Read file content
            contents = await file.read()
            
            # Run AI Engine
            raw_result = process_image_data(contents)
            
            # Extract data
            summary = raw_result.get("plant_summary", {})
            classification = summary.get("classification", {})
            vein = raw_result.get("vein_morphometry", {})
            
            # Base64 Overlay -> Data URL
            overlay_b64 = raw_result.get("overlay_base64")
            overlay_url = None
            if overlay_b64:
                overlay_url = f"data:image/jpeg;base64,{overlay_b64}"
            
            # Format single leaf result for frontend
            formatted_leaf = {
                "leaf_index": idx + 1,
                "filename": file.filename or f"leaf_{idx+1}.jpg",
                "predicted_class": classification.get("predicted_class", "Unknown"),
                "confidence": classification.get("confidence", 0.0),
                "probabilities": classification.get("class_probabilities", {}),
                "lesion_count": summary.get("total_lesion_count", 0),
                "lesion_area_percent": summary.get("avg_lesion_area_percent", 0.0),
                "vein_length_px": vein.get("vein_length_px", 0),
                "vein_density_percent": vein.get("vein_density_percent", 0.0),
                "vein_continuity": vein.get("vein_continuity", 0.0),
                "original_url": None, # Not saving to disk for url, using null or maybe base64 input?
                                      # Frontend might show local file so this is optional.
                "overlay_url": overlay_url
            }
            leaf_results.append(formatted_leaf)
            
            # Accumulate for aggregation
            total_health_score += summary.get("health_score", 0)
            total_confidence += formatted_leaf["confidence"]
            total_lesion_count += formatted_leaf["lesion_count"]
            total_lesion_area += formatted_leaf["lesion_area_percent"]
            
            # Probabilities
            probs = formatted_leaf["probabilities"]
            for cls, score in probs.items():
                all_class_probs[cls] = all_class_probs.get(cls, 0) + score

        # Calculate Averages
        avg_health = total_health_score / num_images
        avg_conf = total_confidence / num_images
        avg_lesion_area = total_lesion_area / num_images
        
        # Normalize Probabilities
        avg_probs = {k: round(v / num_images, 4) for k, v in all_class_probs.items()}
        
        # Final Classification (Max Prob)
        overall_pred_class = "Unknown"
        if avg_probs:
            overall_pred_class = max(avg_probs, key=avg_probs.get)
            
        # Overall Condition Logic
        if overall_pred_class == "Healthy" and avg_conf > 0.4:
            condition = "Healthy"
        elif overall_pred_class != "Healthy" and overall_pred_class != "Unknown":
            condition = "Diseased"
        else:
            condition = "Uncertain"

        # Construct Final Response (Legacy/ScanModel Compatible)
        response_data = {
            "status": "success",
            "scan_id": int(datetime.now().timestamp()), # Generating a temporary ID
            "timestamp": datetime.now().isoformat(),
            "plant_type": plant_type,
            "summary": {
                "health_score": round(avg_health, 1),
                "condition": condition,
                "predicted_class": overall_pred_class,
                "confidence": round(avg_conf, 3),
                "class_probabilities": avg_probs,
                "avg_lesion_area_percent": round(avg_lesion_area, 2),
                "total_lesion_count": total_lesion_count
            },
            "leaves": leaf_results,
            "interpretation": {
                "classification_note": "Based solely on CNN deep learning analysis",
                "health_score_note": "Heuristic indicator (0-100), not a biological diagnosis",
                "vein_note": "Vein analysis is for visual support only, not diagnostic"
            }
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
