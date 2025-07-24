import os
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# -------- Konfigurasi sederhana --------
MODEL_PATH = os.getenv("MODEL_PATH", "model/soilsense_model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)   # dipanggil sekali saja

app = FastAPI(title="SoilSense C-Organic API (.h5 single-file)")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Terima CSV (FormData), kembalikan %C-organik.
    """
    try:
        df = pd.read_csv(file.file)
        X = df.select_dtypes(include=["number"])   # drop kolom waktu jika ada
        y_pred = MODEL.predict(X)[0][0]            # asumsi output skalar
        return {"c_org_pct": float(y_pred)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})