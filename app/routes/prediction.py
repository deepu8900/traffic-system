from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.predictor import predict_traffic_jam

router = APIRouter()

class PredictionRequest(BaseModel):
    vehicle_count: int
    density: float
    speed_kmph: float
    congestion_index: float
    anomaly_detected: bool
    history: List[float] = []

@router.post("/predict")
async def predict(req: PredictionRequest):
    try:
        result = predict_traffic_jam(
            vehicle_count=req.vehicle_count,
            density=req.density,
            speed_kmph=req.speed_kmph,
            congestion_index=req.congestion_index,
            anomaly_detected=req.anomaly_detected,
            history=req.history
        )
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
