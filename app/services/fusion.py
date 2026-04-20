def fuse_traffic_data(traffic_data: dict, cctv_data: dict) -> dict:
    speed = traffic_data.get("speed_kmph", 40)
    congestion_index = traffic_data.get("congestion_index", 0.3)

    if cctv_data:
        density = cctv_data.get("density", 0.0)
        vehicle_count = cctv_data.get("vehicle_count", 0)
        anomaly = cctv_data.get("anomaly_detected", False)
        cctv_weight = 0.45
        traffic_weight = 0.55
        fused_congestion = (density * cctv_weight) + (congestion_index * traffic_weight)
        if anomaly:
            fused_congestion = min(fused_congestion + 0.15, 1.0)
    else:
        vehicle_count = 0
        density = 0.0
        anomaly = False
        fused_congestion = congestion_index

    level = get_congestion_level(fused_congestion)

    return {
        "speed_kmph": speed,
        "congestion_index": round(fused_congestion, 2),
        "congestion_level": level,
        "vehicle_count": vehicle_count,
        "density": density,
        "anomaly_detected": anomaly,
        "source": traffic_data.get("source", "unknown"),
        "location": traffic_data.get("location", {}),
        "timestamp": traffic_data.get("timestamp", "")
    }

def get_congestion_level(index: float) -> str:
    if index < 0.25:
        return "free_flow"
    elif index < 0.5:
        return "moderate"
    elif index < 0.75:
        return "heavy"
    else:
        return "severe"
