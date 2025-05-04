from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from gtts import gTTS
from datetime import datetime
import os
import glob
import mediapipe as mp
import cv2

app = Flask(__name__)
model = load_model("sign_language_mobilenet.h5")

class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def preprocess_frame(data_url):
    _, encoded = data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    img = Image.open(BytesIO(decoded)).convert("RGB")
    img_np = np.array(img)

    results = hand_detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    h, w, _ = img_np.shape

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0].landmark

        x = [lm.x for lm in hand]
        y = [lm.y for lm in hand]

        x_min, x_max = int(min(x) * w), int(max(x) * w)
        y_min, y_max = int(min(y) * h), int(max(y) * h)

        pad_x = int((x_max - x_min) * 0.3)
        pad_y = int((y_max - y_min) * 0.3)

        x_min = max(x_min - pad_x, 0)
        y_min = max(y_min - pad_y, 0)
        x_max = min(x_max + pad_x, w)
        y_max = min(y_max + pad_y, h)

        hand_crop = img_np[y_min:y_max, x_min:x_max]
    else:
        hand_crop = img_np

    hand_resized = cv2.resize(hand_crop, (224, 224))
    return np.expand_dims(hand_resized / 255.0, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    try:
        img_array = preprocess_frame(image_data)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({'prediction': 'Error', 'audio': ''})

    for file in glob.glob("static/prediction_*.mp3"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    audio_filename = f"prediction_{timestamp}.mp3"
    audio_path = os.path.join("static", audio_filename)

    tts = gTTS(text=predicted_class)
    tts.save(audio_path)

    return jsonify({
        'prediction': predicted_class,
        'audio': f"/static/{audio_filename}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)