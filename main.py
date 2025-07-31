from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model("antifungal_peptide_model.h5")

app = FastAPI()

class PeptideInput(BaseModel):
    sequence: str

def encode_sequence(seq):
    aa_dict = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    encoded = [aa_dict.get(a.upper(), 0) for a in seq]
    maxlen = 50
    padded = encoded[:maxlen] + [0] * (maxlen - len(encoded))
    return np.array([padded])

@app.get("/")
def root():
    return {"message": "Antifungal peptide predictor is live!"}

@app.post("/predict")
def predict(peptide: PeptideInput):
    x = encode_sequence(peptide.sequence)
    prediction = model.predict(x)
    return {
        "sequence": peptide.sequence,
        "prediction": float(prediction[0][0])
    }
