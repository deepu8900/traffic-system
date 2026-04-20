from fastapi import APIRouter, Query
from app.services.traffic_fetcher import get_traffic_data
from app.services.fusion import fuse_traffic_data

router = APIRouter()

@router.get("/traffic-data")
async def traffic_data(lat: float = Query(28.6139), lng: float = Query(77.2090)):
    raw = get_traffic_data(lat, lng)
    fused = fuse_traffic_data(raw, None)
    return {"status": "success", "data": fused}
