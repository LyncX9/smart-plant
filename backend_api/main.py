
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

# Import our engine
from model_engine import process_image_data, load_model, MODEL_PATH

# Import database
from database import init_db, get_db, is_db_available
from models import ScanHistory

app = FastAPI(title="SmartPlant Vision API", version="2.3")

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
    # Load ML model
    if os.path.exists(MODEL_PATH):
        load_model(MODEL_PATH)
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found!")
    
    # Initialize database
    init_db()

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "service": "SmartPlant Vision API",
        "database": "connected" if is_db_available() else "not configured"
    }

@app.post("/analyze")
async def analyze_image(
    images: List[UploadFile] = File(alias="images[]"),
    plant_type: str = Form("rice"),
    db: Session = Depends(get_db)
):
    try:
        leaf_results = []
        
        # Aggregation variables
        total_health_score = 0
        total_confidence = 0
        total_lesion_count = 0
        total_lesion_area = 0
        all_class_probs = {}
        
        num_images = len(images)
        if num_images == 0:
             raise HTTPException(status_code=400, detail="No images provided")

        # Process each image
        for idx, file in enumerate(images):
            contents = await file.read()
            raw_result = process_image_data(contents)
            
            summary = raw_result.get("plant_summary", {})
            classification = summary.get("classification", {})
            vein = raw_result.get("vein_morphometry", {})
            
            overlay_b64 = raw_result.get("overlay_base64")
            overlay_url = f"data:image/jpeg;base64,{overlay_b64}" if overlay_b64 else None
            
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
                "original_url": None,
                "overlay_url": overlay_url
            }
            leaf_results.append(formatted_leaf)
            
            total_health_score += summary.get("health_score", 0)
            total_confidence += formatted_leaf["confidence"]
            total_lesion_count += formatted_leaf["lesion_count"]
            total_lesion_area += formatted_leaf["lesion_area_percent"]
            
            probs = formatted_leaf["probabilities"]
            for cls, score in probs.items():
                all_class_probs[cls] = all_class_probs.get(cls, 0) + score

        # Calculate Averages
        avg_health = total_health_score / num_images
        avg_conf = total_confidence / num_images
        avg_lesion_area = total_lesion_area / num_images
        avg_probs = {k: round(v / num_images, 4) for k, v in all_class_probs.items()}
        
        overall_pred_class = "Unknown"
        if avg_probs:
            overall_pred_class = max(avg_probs, key=avg_probs.get)
            
        if overall_pred_class == "Healthy" and avg_conf > 0.25:
            condition = "Healthy"
        elif overall_pred_class != "Healthy" and overall_pred_class != "Unknown":
            condition = "Diseased"
        else:
            condition = "Uncertain"

        # Generate scan ID
        scan_id = int(datetime.now().timestamp())
        
        # Save to database if available
        if db is not None:
            try:
                scan_record = ScanHistory(
                    plant_type=plant_type,
                    condition=condition,
                    predicted_class=overall_pred_class,
                    confidence=round(avg_conf, 4),
                    health_score=round(avg_health, 1),
                    total_lesion_count=total_lesion_count,
                    avg_lesion_area_percent=round(avg_lesion_area, 2),
                    leaves_count=num_images,
                    leaves_json=json.dumps(leaf_results)
                )
                db.add(scan_record)
                db.commit()
                db.refresh(scan_record)
                scan_id = scan_record.id
            except Exception as e:
                print(f"Failed to save scan to database: {e}")
                db.rollback()

        # Construct Response
        response_data = {
            "status": "success",
            "scan_id": scan_id,
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


# ============================================================================
# HISTORY ENDPOINTS
# ============================================================================

@app.get("/history")
async def get_history(
    plant_type: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get scan history with optional filters."""
    if db is None:
        # Database not configured - return empty list
        return JSONResponse(content={"data": [], "page": page, "per_page": per_page, "total": 0})
    
    try:
        query = db.query(ScanHistory)
        
        # Apply filters
        if plant_type:
            query = query.filter(ScanHistory.plant_type == plant_type)
        if condition:
            query = query.filter(ScanHistory.condition == condition)
        
        # Get total count
        total = query.count()
        
        # Paginate and order by newest first
        scans = query.order_by(ScanHistory.timestamp.desc()) \
                     .offset((page - 1) * per_page) \
                     .limit(per_page) \
                     .all()
        
        return JSONResponse(content={
            "data": [scan.to_dict() for scan in scans],
            "page": page,
            "per_page": per_page,
            "total": total
        })
    except Exception as e:
        print(f"Error fetching history: {e}")
        return JSONResponse(content={"data": [], "page": page, "per_page": per_page, "total": 0})


@app.get("/history/{scan_id}")
async def get_scan_detail(scan_id: int, db: Session = Depends(get_db)):
    """Get detailed scan result by ID."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    scan = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    # Build response matching frontend expectations
    detail = scan.to_detail_dict()
    
    return JSONResponse(content={
        "status": "success",
        "data": {
            "scan_id": detail["id"],
            "timestamp": detail["created_at"],
            "plant_type": detail["plant_type"],
            "summary": {
                "health_score": detail["health_score"],
                "condition": detail["condition"],
                "predicted_class": detail["predicted_class"],
                "confidence": detail["confidence"],
                "avg_lesion_area_percent": detail["avg_lesion_area_percent"],
                "total_lesion_count": detail["total_lesion_count"],
                "class_probabilities": {}
            },
            "leaves": detail["leaves"],
            "interpretation": {
                "classification_note": "Based solely on CNN deep learning analysis",
                "health_score_note": "Heuristic indicator (0-100), not a biological diagnosis",
                "vein_note": "Vein analysis is for visual support only, not diagnostic"
            }
        }
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
