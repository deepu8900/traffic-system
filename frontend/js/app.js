const API = "http://localhost:8000";

let cctvData = null;
let trafficData = null;

function clock() {
    const el = document.getElementById("clock");
    const tick = () => {
        const now = new Date();
        el.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
    };
    tick();
    setInterval(tick, 1000);
}

function toast(msg, type = "info") {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.className = "toast" + (type === "error" ? " error" : "");
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.className = "toast hidden"; }, 3500);
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (loading) {
        btn.dataset.orig = btn.textContent;
        btn.textContent = "PROCESSING...";
        btn.disabled = true;
        btn.style.opacity = "0.6";
    } else {
        btn.textContent = btn.dataset.orig || btn.textContent;
        btn.disabled = false;
        btn.style.opacity = "1";
    }
}

function riskColor(pct) {
    if (pct < 30) return "#2ecc71";
    if (pct < 55) return "#e8c840";
    if (pct < 75) return "#e88030";
    return "#e03030";
}

function animateRiskBar(id, pctId, value) {
    const fill = document.getElementById(id);
    const label = document.getElementById(pctId);
    const pct = Math.round(value * 100);
    setTimeout(() => {
        fill.style.width = pct + "%";
        fill.style.background = riskColor(pct);
    }, 80);
    label.textContent = pct + "%";
}

const dropZone = document.getElementById("drop-zone");
const videoInput = document.getElementById("video-input");

dropZone.addEventListener("click", () => videoInput.click());

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length) {
        videoInput.files = files;
        dropZone.querySelector(".upload-hint").textContent = files[0].name;
    }
});

videoInput.addEventListener("change", () => {
    if (videoInput.files.length) {
        dropZone.querySelector(".upload-hint").textContent = videoInput.files[0].name;
    }
});

document.getElementById("analyze-btn").addEventListener("click", async () => {
    if (!videoInput.files.length) {
        toast("Select a video file first.", "error");
        return;
    }

    setLoading("analyze-btn", true);
    const form = new FormData();
    form.append("file", videoInput.files[0]);

    try {
        const res = await fetch(`${API}/upload-video`, { method: "POST", body: form });
        if (!res.ok) throw new Error("Server error " + res.status);
        const data = await res.json();
        const a = data.analysis;
        cctvData = a;

        document.getElementById("m-vehicles").textContent = a.vehicle_count;
        document.getElementById("m-density").textContent = (a.density * 100).toFixed(0) + "%";
        document.getElementById("m-stopped").textContent = a.stopped_vehicles;

        const anomEl = document.getElementById("m-anomaly");
        anomEl.textContent = a.anomaly_detected ? "YES" : "NO";
        anomEl.style.color = a.anomaly_detected ? "var(--red)" : "var(--green)";

        document.getElementById("density-fill").style.width = (a.density * 100) + "%";
        document.getElementById("cctv-result").classList.remove("hidden");
        document.getElementById("manual-form").classList.remove("hidden");
        toast("Video analysis complete.");
    } catch (err) {
        toast("Analysis failed: " + err.message, "error");
    } finally {
        setLoading("analyze-btn", false);
    }
});

document.getElementById("traffic-btn").addEventListener("click", async () => {
    const lat = parseFloat(document.getElementById("lat-input").value);
    const lng = parseFloat(document.getElementById("lng-input").value);

    if (isNaN(lat) || isNaN(lng)) {
        toast("Enter valid coordinates.", "error");
        return;
    }

    setLoading("traffic-btn", true);

    try {
        const res = await fetch(`${API}/traffic-data?lat=${lat}&lng=${lng}`);
        if (!res.ok) throw new Error("Server error " + res.status);
        const json = await res.json();
        const d = json.data;
        trafficData = d;

        document.getElementById("t-speed").textContent = d.speed_kmph;
        document.getElementById("t-congestion").textContent = (d.congestion_index * 100).toFixed(0) + "%";
        document.getElementById("t-delay").textContent = d.delay_seconds;

        const lvlEl = document.getElementById("t-level");
        const lvlMap = { free_flow: "FREE FLOW", moderate: "MODERATE", heavy: "HEAVY", severe: "SEVERE" };
        lvlEl.textContent = lvlMap[d.congestion_level] || d.congestion_level;
        lvlEl.className = "metric-value level-badge " + d.congestion_level;

        const src = d.source === "google_maps" ? "SOURCE: GOOGLE MAPS API" : "SOURCE: SIMULATED DATA ENGINE";
        document.getElementById("t-source").textContent = src;
        document.getElementById("traffic-result").classList.remove("hidden");
        document.getElementById("manual-form").classList.remove("hidden");
        toast("Traffic data fetched.");
    } catch (err) {
        toast("Fetch failed: " + err.message, "error");
    } finally {
        setLoading("traffic-btn", false);
    }
});

document.getElementById("autofill-btn").addEventListener("click", () => {
    const source = cctvData || trafficData;
    if (!source) {
        toast("Run CCTV analysis or fetch traffic data first.", "error");
        return;
    }
    if (cctvData) {
        document.getElementById("p-vehicles").value = cctvData.vehicle_count;
        document.getElementById("p-density").value = cctvData.density;
        document.getElementById("p-anomaly").checked = cctvData.anomaly_detected;
    }
    if (trafficData) {
        document.getElementById("p-speed").value = trafficData.speed_kmph;
        document.getElementById("p-ci").value = trafficData.congestion_index;
    }
    document.getElementById("manual-form").classList.remove("hidden");
    toast("Fields populated from latest data.");
});

document.getElementById("predict-btn").addEventListener("click", async () => {
    document.getElementById("manual-form").classList.remove("hidden");

    const payload = {
        vehicle_count: parseInt(document.getElementById("p-vehicles").value) || 0,
        density: parseFloat(document.getElementById("p-density").value) || 0,
        speed_kmph: parseFloat(document.getElementById("p-speed").value) || 30,
        congestion_index: parseFloat(document.getElementById("p-ci").value) || 0,
        anomaly_detected: document.getElementById("p-anomaly").checked,
        history: []
    };

    setLoading("predict-btn", true);

    try {
        const res = await fetch(`${API}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error("Server error " + res.status);
        const json = await res.json();
        const p = json.prediction;

        animateRiskBar("r15", "r15-pct", p.jam_probability_15min);
        animateRiskBar("r30", "r30-pct", p.jam_probability_30min);
        animateRiskBar("r45", "r45-pct", p.jam_probability_45min);

        const mTag = document.getElementById("p-model-tag");
        mTag.textContent = p.model_used === "lstm_keras" ? "[ LSTM MODEL ]" : "[ RULE-BASED ]";

        const risk = p.overall_risk;
        const verdict = document.getElementById("risk-verdict");
        if (risk < 0.3) {
            verdict.textContent = "▸ CONDITIONS NOMINAL — LOW JAM RISK IN NEXT 45 MIN";
            verdict.style.borderLeftColor = "var(--green)";
            verdict.style.background = "rgba(46,204,113,0.07)";
        } else if (risk < 0.6) {
            verdict.textContent = "▸ MODERATE RISK — MONITOR CLOSELY, CONGESTION POSSIBLE";
            verdict.style.borderLeftColor = "#e8c840";
            verdict.style.background = "rgba(232,200,64,0.07)";
        } else if (risk < 0.8) {
            verdict.textContent = "▸ HIGH RISK — TRAFFIC JAM LIKELY, REROUTING ADVISED";
            verdict.style.borderLeftColor = "#e88030";
            verdict.style.background = "rgba(232,128,48,0.07)";
        } else {
            verdict.textContent = "▸ CRITICAL — SEVERE JAM IMMINENT, IMMEDIATE ACTION REQUIRED";
            verdict.style.borderLeftColor = "var(--red)";
            verdict.style.background = "rgba(224,48,48,0.09)";
        }
        verdict.classList.add("show");

        document.getElementById("predict-result").classList.remove("hidden");
        toast("Prediction complete.");
    } catch (err) {
        toast("Prediction failed: " + err.message, "error");
    } finally {
        setLoading("predict-btn", false);
    }
});

clock();
document.getElementById("manual-form").classList.remove("hidden");
