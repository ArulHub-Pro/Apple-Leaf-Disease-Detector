from flask import Flask, render_template, request
import os, numpy as np, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load .keras model
model = load_model('../model/apple_disease_model.keras')

with open('../suggestions/suggestions.json') as f:
    suggestions = json.load(f)

classes = ['Apple_black_rot', 'Apple_cedar_rust', 'Apple_scab']
threshold = 0.7

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    confidence = max(pred)
    if confidence < threshold:
        return 'Healthy', confidence
    return classes[np.argmax(pred)], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)
        disease, confidence = predict(path)
        data = suggestions[disease]
        return render_template('index.html', path=path, disease=disease, confidence=round(confidence*100, 2), data=data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
