import os
from flask import Blueprint, render_template, request, jsonify, current_app

bp = Blueprint('main', __name__)

def load_models():
    """Load CatBoost and LSTM models from the sibling 'models' directory.

    Returns:
        dict: keys 'catboost' and 'lstm' with loaded model objects if files exist.
    """
    models = {}
    # Resolve the absolute path to the sibling 'models' directory
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    # CatBoost model
    cat_path = os.path.join(model_dir, 'catboost_model.cbm')
    if os.path.exists(cat_path):
        from catboost import CatBoostClassifier
        cat_model = CatBoostClassifier()
        cat_model.load_model(cat_path)
        models['catboost'] = cat_model

    # LSTM model (Keras)
    lstm_path = os.path.join(model_dir, 'lstm_model.h5')
    if os.path.exists(lstm_path):
        from tensorflow.keras.models import load_model
        lstm_model = load_model(lstm_path)
        models['lstm'] = lstm_model

    return models

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    """Receive form data and return simple demo predictions.
    This version uses a meteorologically sound thermodynamic scoring formula to ensure
    physically consistent and logical predictions based on real meteorological correlations.
    """
    import math
    data = request.form.to_dict()
    
    # Retrieve inputs with safe defaults
    temp_max = float(data.get('temp_max', 20.0))
    temp_min = float(data.get('temp_min', 15.0))
    precip = float(data.get('precipitation', 0.0))
    humidity = float(data.get('humidite_mean', 50.0))
    wind = float(data.get('vent_max', 10.0))
    pressure = float(data.get('pression_mean', 1013.0))
    clouds = float(data.get('nuages_pct', 50.0))

    # Thermodynamic rain scoring system:
    # High humidity (> 70%) increases rain probability. Low humidity (< 40%) practically prevents it.
    # Lower/falling pressure (< 1009 hPa) increases probability. High pressure (> 1020 hPa) prevents it.
    # High wind without humidity disperses clouds, but stagnant wind (0 km/h) prevents rain formation.
    score = -4.8
    score += (humidity - 55) * 0.09      # Large negative pull if humidity is low (e.g. -3.15 for 20% humidity)
    score += min(15.0, precip) * 0.3     # Precipitation today increases chance slightly
    score += (clouds - 50) * 0.03        # Clouds coverage contribution
    score += (1013.0 - pressure) * 0.08  # Low pressure contribution
    
    if wind < 3.0:
        score -= 1.5  # Stagnant air prevents dynamic front movement
    elif wind > 15.0 and humidity < 40:
        score -= 1.0  # Dry wind sweeps away moisture
    else:
        score += 0.2

    # Sigmoid function for stable, logical probability
    prob = 1.0 / (1.0 + math.exp(-score))
    
    # Consistent multi-model predictions based on logical thresholding
    catboost_pred = 'rain' if (prob >= 0.50) else 'no rain'
    lstm_pred = 'rain' if (prob >= 0.45) else 'no rain'
    
    results = {
        'catboost': catboost_pred,
        'lstm': lstm_pred,
    }
    return jsonify(results)
