from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load model
model = load_model("antifungal_peptide_model.h5")

# Amino acid encoding
aa_to_int = {
    'a': 1, 'r': 2, 'n': 3, 'd': 4, 'c': 5, 'q': 6, 'e': 7, 'g': 8, 'h': 9, 'i': 10,
    'l': 11, 'k': 12, 'm': 13, 'f': 14, 'p': 15, 's': 16, 't': 17, 'w': 18, 'y': 19, 'v': 20
}

max_length = 40

def encode_sequence(seq):
    seq = seq.lower()
    encoded = [aa_to_int.get(aa, 0) for aa in seq]
    return pad_sequences([encoded], maxlen=max_length, padding='post')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    sequence = ""
    if request.method == "POST":
        sequence = request.form["sequence"]
        if sequence:
            encoded = encode_sequence(sequence)
            prob = model.predict(encoded)[0][0]
            prediction = f"{prob:.4f}"
    return render_template("index.html", prediction=prediction, sequence=sequence)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
