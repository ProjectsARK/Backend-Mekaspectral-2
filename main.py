import os
from datetime import datetime
from typing import List, Union

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# ───── Konfigurasi ──────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH", "best_model_ann_raw_SOM.h5")
MODEL_VERSION = os.getenv("MODEL_VERSION", "ann_raw_v1.0")

# Muat model sekali saja saat container start
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ───── Aplikasi FastAPI ────────────────────────────────────────
app = FastAPI(
    title   = "SoilSense C-Organic API",
    version = MODEL_VERSION
)

# ——— health-check sederhana (dipakai LB / Codespaces) ———
@app.get("/health")
def health():
    """Return 200 jika model berhasil dimuat."""
    return {
        "status"        : "ok",
        "model_version" : MODEL_VERSION,
        "server_time"   : datetime.utcnow().isoformat() + "Z"
    }

# ——— endpoint utama inferensi ————————————
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Terima file CSV via multipart/form-data dan kembalikan
    prediksi %C-organik.

    • Jika CSV berisi 1 baris ⇒ { "c_org_pct": <float> }
    • Jika >1 baris        ⇒ { "predictions": [float, …] }
    """
    try:
        # Baca CSV langsung dari stream
        df = pd.read_csv(file.file)

        # Buang kolom target jika ada
        if "SOM (%)" in df.columns:
            df = df.drop(columns=["SOM (%)"])

        # Hapus fitur 485nm yang memang tidak dipakai
        if "485nm" in df.columns:
            df = df.drop(columns=["485nm"])

        # Pastikan hanya kolom numerik yang dipakai
        X = df.select_dtypes(include=["number"])

        # Jalankan prediksi
        y_pred: List[float] = MODEL.predict(X).ravel().tolist()

        # Bangun payload adaptif
        if len(y_pred) == 1:
            return {"c_org_pct": y_pred[0]}
        return {"predictions": y_pred}

    except Exception as e:
        # Pesan error ramah untuk klien
        return JSONResponse(
            status_code = 400,
            content     = {"error": str(e)}
        )
