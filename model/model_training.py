from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

model = None
scaler = None
if os.path.exists('model.pkl'):
    model = joblib.load('model.pkl')
if os.path.exists('scaler.pkl'):
    scaler = joblib.load('scaler.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    if model is None:
        return jsonify({'error': 'model not found on server'}), 500
    prediction = model.predict(features)[0]
    # ensure JSON-serializable output
    if hasattr(prediction, 'tolist'):
        out = prediction.tolist()
    elif np.isscalar(prediction):
        out = int(prediction)
    else:
        out = prediction
    return jsonify({'prediction': out})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
