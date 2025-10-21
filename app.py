from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = tf.saved_model.load("saved_model")

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [l.strip() for l in f.readlines()]

def classify_array(image_array):
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    prediction_dict = model(data)
    if isinstance(prediction_dict, dict):
        for key in ['sequential_3', 'outputs', 'output_0', 'predictions', 'dense', 'Identity']:
            if key in prediction_dict:
                pred = prediction_dict[key]
                break
        else:
            pred = list(prediction_dict.values())[0]
    else:
        pred = prediction_dict
    pred = pred.numpy() if hasattr(pred, "numpy") else np.array(pred)
    idx = int(np.argmax(pred, axis=1)[0])
    name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
    conf = float(pred[0][idx])
    return name, conf

def map_to_binary(label):
    l = label.lower()
    if "organic" in l:
        return "Organic"
    if "inorganic" in l:
        return "Inorganic"
    return label

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    files = request.files.getlist("images")
    results = []
    for f in files:
        img = Image.open(f.stream).convert("RGB")
        img_resized = ImageOps.fit(img, (224,224), Image.Resampling.LANCZOS)
        arr = np.asarray(img_resized)
        name, conf = classify_array(arr)
        results.append({
            "filename": f.filename,
            "prediction": name,
            "binary": map_to_binary(name),
            "confidence": conf
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
