import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from tensorflow import keras

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'tunisie_meteo_final.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Features and target
    feature_cols = [c for c in df.columns if c not in ['pluie_demain_bin', 'pluie_demain_mm', 'date']]
    X = df[feature_cols]
    y = df['pluie_demain_bin']
    # Chronological split matching training script
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'].dt.year >= 2024]
    X_test = test_df[feature_cols]
    y_test = test_df['pluie_demain_bin']
    return X_test, y_test

def load_catboost_model():
    model_path = os.path.join(MODEL_DIR, 'catboost_binary_rain.cbm')
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def load_keras_model():
    model_path = os.path.join(MODEL_DIR, 'nn_classifier.keras')
    model = keras.models.load_model(model_path)
    return model

def encode_categorical(df, reference_df=None):
    # Encode object columns as integers using factorize based on training data
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        if reference_df is not None and col in reference_df.columns:
            # Use mapping from reference (training) dataframe
            uniques = pd.factorize(reference_df[col])[1]
            mapping = {v:k for k,v in enumerate(uniques)}
            df_enc[col] = df_enc[col].map(mapping).fillna(-1).astype(int)
        else:
            df_enc[col], _ = pd.factorize(df_enc[col])
    return df_enc

def evaluate_model(name, preds, y_true):
    auc_score = roc_auc_score(y_true, preds)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, preds)
    pr_auc = auc(pr_recall, pr_precision)
    pred_labels = (preds >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels, zero_division=0)
    report = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred_labels)
    metrics = {
        'name': name,
        'roc_auc': auc_score,
        'pr_auc': pr_auc,
        'accuracy': acc,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    return metrics

def main():
    X_test, y_test = load_data()
    # CatBoost evaluation (expects original features)
    cat_model = load_catboost_model()
    cat_preds = cat_model.predict_proba(X_test)[:, 1]
    cat_metrics = evaluate_model('catboost', cat_preds, y_test)
    # Keras evaluation (needs same encoding as training)
    # Load training data to obtain encoding mapping
    df_train = pd.read_csv(DATA_PATH)
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train = df_train[df_train['date'].dt.year <= 2022]
    X_train = df_train[[c for c in df_train.columns if c not in ['pluie_demain_bin', 'pluie_demain_mm', 'date']]]
    X_test_enc = encode_categorical(X_test, reference_df=X_train)
    keras_model = load_keras_model()
    keras_preds = keras_model.predict(X_test_enc).ravel()
    keras_metrics = evaluate_model('keras', keras_preds, y_test)
    # Save metrics JSON
    all_metrics = {'catboost': cat_metrics, 'keras': keras_metrics}
    metrics_path = os.path.join(RESULTS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
