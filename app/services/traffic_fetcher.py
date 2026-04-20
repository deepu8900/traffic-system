import os
import random
import requests
from datetime import datetime

GMAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

def fetch_from_google_maps(lat, lng):
    origin = f"{lat},{lng}"
    dest_lat = lat + 0.02
    dest_lng = lng + 0.02
    destination = f"{dest_lat},{dest_lng}"

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "departure_time": "now",
        "traffic_model": "best_guess",
        "key": GMAPS_API_KEY
    }

    resp = requests.get(url, params=params, timeout=5)
    data = resp.json()

    if data.get("status") != "OK":
        return None

    leg = data["routes"][0]["legs"][0]
    duration_sec = leg["duration"]["value"]
    duration_traffic_sec = leg.get("duration_in_traffic", {}).get("value", duration_sec)
    distance_m = leg["distance"]["value"]

    speed_kmph = (distance_m / duration_traffic_sec) * 3.6 if duration_traffic_sec > 0 else 30
    delay_ratio = duration_traffic_sec / duration_sec if duration_sec > 0 else 1.0
    congestion_index = min((delay_ratio - 1.0) * 2.5, 1.0)

    return {
        "speed_kmph": round(speed_kmph, 1),
        "congestion_index": round(max(congestion_index, 0), 2),
        "delay_seconds": max(0, duration_traffic_sec - duration_sec),
        "source": "google_maps"
    }

def simulate_traffic_data(lat, lng):
    hour = datetime.now().hour
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)

    if is_peak:
        speed = random.uniform(8, 25)
        congestion = random.uniform(0.6, 0.95)
    else:
        speed = random.uniform(30, 65)
        congestion = random.uniform(0.05, 0.4)

    return {
        "speed_kmph": round(speed, 1),
        "congestion_index": round(congestion, 2),
        "delay_seconds": int(congestion * 600),
        "source": "simulated",
        "location": {"lat": lat, "lng": lng},
        "timestamp": datetime.now().isoformat()
    }

def get_traffic_data(lat: float, lng: float):
    if GMAPS_API_KEY:
        try:
            result = fetch_from_google_maps(lat, lng)
            if result:
                result["location"] = {"lat": lat, "lng": lng}
                result["timestamp"] = datetime.now().isoformat()
                return result
        except Exception:
            pass

    return simulate_traffic_data(lat, lng)
