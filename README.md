# Hybrid Predictive Traffic Intelligence System (HPTIS)

A full-stack AI system combining CCTV video analysis, live traffic data, and LSTM-based predictions to forecast traffic jams.

---

## Folder Structure

```
traffic_system/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── routes/
│   │   ├── video.py             # POST /upload-video
│   │   ├── traffic.py           # GET /traffic-data
│   │   └── prediction.py        # POST /predict
│   └── services/
│       ├── yolo_analyzer.py     # CCTV frame analysis
│       ├── traffic_fetcher.py   # Google Maps / simulated data
│       ├── fusion.py            # Combines CCTV + traffic data
│       └── predictor.py         # LSTM model inference
├── models/
│   └── traffic_lstm.h5          # Trained Keras model (generate or train)
├── scripts/
│   ├── train_model.py           # Full LSTM training script (run on Kaggle GPU)
│   └── generate_dummy_model.py  # Quick test model for local dev
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── data/
│   └── uploads/                 # Auto-created on first video upload
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── requirements.txt
└── README.md
```

---

## Setup Instructions (VS Code)

### Step 1 — Open the project
```
File → Open Folder → select traffic_system/
```

### Step 2 — Create a virtual environment
Open the integrated terminal (Ctrl + ` ) and run:
```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate a local test model
```bash
cd scripts
python generate_dummy_model.py
cd ..
```
This creates `models/traffic_lstm.h5` for local testing.
For production accuracy, run `train_model.py` on a Kaggle GPU notebook instead.

### Step 5 — (Optional) Add YOLO weights for real video analysis
Download YOLOv3 files and place them in `models/`:
- `yolov3.weights` — https://pjreddie.com/media/files/yolov3.weights
- `yolov3.cfg` — https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
- `coco.names` — https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

Without these files the system uses built-in simulation (still fully functional).

### Step 6 — (Optional) Set Google Maps API key
```bash
# Windows
set GOOGLE_MAPS_API_KEY=your_key_here

# Mac/Linux
export GOOGLE_MAPS_API_KEY=your_key_here
```
Without a key the system uses simulated traffic data.

### Step 7 — Run the server

Option A — VS Code debugger:
Press `F5` (uses `.vscode/launch.json`)

Option B — Terminal:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 8 — Open the UI
Navigate to: http://localhost:8000

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/upload-video` | Upload a video file for CCTV analysis |
| GET | `/traffic-data?lat=&lng=` | Fetch traffic data for coordinates |
| POST | `/predict` | Run LSTM prediction for jam probability |
| GET | `/docs` | FastAPI auto-generated Swagger UI |

---

## Training on Kaggle

1. Upload `scripts/train_model.py` to a Kaggle notebook
2. Enable GPU accelerator (P100 recommended)
3. Run the script — training completes in ~5 minutes
4. Download `traffic_lstm.h5` from the output
5. Place it in the local `models/` folder

---

## Notes

- The LSTM model accepts sequences of 10 timesteps with 5 features each
- Prediction outputs jam probability for 15, 30, and 45 minutes ahead
- If the `.h5` model file is missing, the system automatically falls back to a rule-based predictor
- YOLO detection uses OpenCV's DNN module (TensorFlow-compatible weights format)
