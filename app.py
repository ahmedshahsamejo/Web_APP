from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import io
import numpy as np
from PIL import Image
from keras.models import load_model
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def ret_fresh(res):
    threshold_fresh = 0.90
    threshold_medium = 0.50
    if res > threshold_fresh:
        return "The item is VERY FRESH!"
    elif threshold_fresh > res > threshold_medium:
        return "The item is FRESH"
    else:
        return "The item is NOT FRESH"

def pre_proc_img(image_data):
    byte_stream = io.BytesIO()
    image_data = image_data.convert('RGB')
    image_data.save(byte_stream, format='JPEG')
    image_bytes = byte_stream.getvalue()

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_rotten_vs_fresh(image_path):
    model = load_model('C:/Users/PMLS/Desktop/Web_App/rottenvsfresh98pval.h5')  # Update with the actual path
    prediction = model.predict(pre_proc_img(image_path))
    return prediction[0][0]

def ident_type(img):
    model = models.mobilenet_v2(weights=None)
    num_classes = 36
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('C:/Users/PMLS/Desktop/Web_App/modelforclass.pth', map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
                    'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
                    'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                    'watermelon']
    with torch.no_grad():
        img = transform(img)
        img = img.unsqueeze(0)
        output = model(img)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    img = Image.open(image)
    is_fresh = 1 - evaluate_rotten_vs_fresh(img)
    pred_type = ident_type(img)
    if is_fresh == 0.0:
        result = {
            'Prediction': str(is_fresh),
            'Freshness': ret_fresh(is_fresh),
            'Type': ' '
        }
    else:
        
        result = {
        'Prediction': str(is_fresh),
        'Freshness': ret_fresh(is_fresh),
        'Type': pred_type
    }

    return render_template('result.html', result=result)
@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000)
