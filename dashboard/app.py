from flask import Flask, render_template, jsonify
import torch
import json
import os
import base64
from models.global_model import HeartDiseaseModel

app = Flask(__name__)

def load_global_model():
    """Load the trained global model"""
    model = HeartDiseaseModel(input_size=13)
    model_path = os.path.join('results', 'global_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    return model

def load_results():
    """Load training results"""
    results_path = os.path.join('results', 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

@app.route('/')
def index():
    """Main dashboard page"""
    results = load_results()

    # Get accuracy data
    rounds = []
    global_accuracies = []
    client1_acc = []
    client2_acc = []
    client3_acc = []

    for round_key, round_data in results.items():
        round_num = int(round_key.split('_')[1])
        rounds.append(round_num)
        global_accuracies.append(
            round(round_data['global_accuracy'] * 100, 2))
        client1_acc.append(
            round(round_data['client_accuracies'][0] * 100, 2))
        client2_acc.append(
            round(round_data['client_accuracies'][1] * 100, 2))
        client3_acc.append(
            round(round_data['client_accuracies'][2] * 100, 2))

    # Get final accuracy
    final_accuracy = global_accuracies[-1] if global_accuracies else 0

    # Load images
    shap_summary = image_to_base64('results/shap_summary.png')
    shap_bar = image_to_base64('results/shap_bar.png')
    lime1 = image_to_base64('results/lime_explanation_1.png')
    lime2 = image_to_base64('results/lime_explanation_2.png')
    lime3 = image_to_base64('results/lime_explanation_3.png')
    training_plot = image_to_base64('results/training_results.png')

    return render_template('index.html',
        rounds=rounds,
        global_accuracies=global_accuracies,
        client1_acc=client1_acc,
        client2_acc=client2_acc,
        client3_acc=client3_acc,
        final_accuracy=final_accuracy,
        shap_summary=shap_summary,
        shap_bar=shap_bar,
        lime1=lime1,
        lime2=lime2,
        lime3=lime3,
        training_plot=training_plot
    )

@app.route('/predict', methods=['GET'])
def predict():
    """Sample prediction endpoint"""
    model = load_global_model()
    import torch
    sample = torch.randn(1, 13)
    with torch.no_grad():
        output = model(sample).item()
    prediction = "Heart Disease Detected" if output > 0.5 else "No Heart Disease"
    confidence = round(output * 100, 2)
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    print("Starting VitaFed AI Dashboard...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True)