import os
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
from tensorflow import keras

# Paths (relative to project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'tunisie_meteo_final.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    feature_cols = [c for c in df.columns if c not in ['pluie_demain_bin', 'pluie_demain_mm', 'date']]
    return df[feature_cols], df['pluie_demain_bin']

def encode_categorical(df):
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col], _ = pd.factorize(df_enc[col])
    return df_enc

def shap_catboost():
    cat_path = os.path.join(MODEL_DIR, 'catboost_binary_rain.cbm')
    model = CatBoostClassifier()
    model.load_model(cat_path)
    X, _ = load_data()
    X_enc = encode_categorical(X)
    # Use a sample for speed
    sample = X_enc.sample(min(2000, len(X_enc)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    shap_path = os.path.join(RESULTS_DIR, 'shap_catboost.png')
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    return shap_path

def permutation_importance_lstm():
    keras_path = os.path.join(MODEL_DIR, 'nn_classifier.keras')
    model = keras.models.load_model(keras_path)
    # Wrap Keras model to provide predict_proba for sklearn scorer
    from sklearn.base import BaseEstimator, ClassifierMixin
    class KerasWrapper(BaseEstimator, ClassifierMixin):
        _estimator_type = "classifier"
        """Thin wrapper making a Keras model look like a scikit-learn classifier.
        Implements ``predict_proba`` returning a two‑column array required by sklearn
        and ``predict`` for binary decisions. Inherits from ``BaseEstimator``
        so ``sklearn.utils._tags.get_tags`` works without error.
        """
        def fit(self, X, y=None):
            # No training needed; model is already trained
            return self
        def __init__(self, keras_model):
            self.model = keras_model
        def predict_proba(self, X):
            probs = self.model.predict(X).ravel()
            return np.column_stack([1 - probs, probs])
        def predict(self, X):
            probs = self.model.predict(X).ravel()
            return (probs >= 0.5).astype(int)
    estimator = KerasWrapper(model)
    estimator._estimator_type = "classifier"
    X, y = load_data()
    X_enc = encode_categorical(X)
    # Use a subset for speed
    X_subset = X_enc.sample(min(2000, len(X_enc)), random_state=42)
    y_subset = y.loc[X_subset.index]
    result = permutation_importance(
        estimator=estimator,
        X=X_subset,
        y=y_subset,
        scoring='accuracy',
        n_repeats=5,
        random_state=42,
        n_jobs=1,
        response_method='predict',
    )
    importances = result.importances_mean
    # Plot
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), X_subset.columns[indices], rotation=90)
    plt.title('Permutation Importance - LSTM (AUC)')
    perm_path = os.path.join(RESULTS_DIR, 'perm_importance_lstm.png')
    plt.tight_layout()
    plt.savefig(perm_path)
    plt.close()
    return perm_path

def main():
    shap_path = shap_catboost()
    perm_path = permutation_importance_lstm()
    summary = {
        'shap_catboost_path': shap_path,
        'permutation_importance_lstm_path': perm_path,
    }
    json_path = os.path.join(RESULTS_DIR, 'interpretability_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Interpretability artifacts saved to {RESULTS_DIR}')

if __name__ == "__main__":
    main()
