import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'tunisie_meteo_final.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'pluie_demain_bin' not in df.columns:
        raise ValueError('Target column "pluie_demain_bin" not found in dataset')
    return df

def split_data(df):
    # Chronological split: train 2009-2022, validation 2023, test 2024-2026
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year <= 2022]
    val_df = df[df['date'].dt.year == 2023]
    test_df = df[df['date'].dt.year >= 2024]
    feature_cols = [c for c in df.columns if c not in ['pluie_demain_bin', 'pluie_demain_mm', 'date']]
    X_train = train_df[feature_cols]
    y_train = train_df['pluie_demain_bin']
    X_val = val_df[feature_cols]
    y_val = val_df['pluie_demain_bin']
    X_test = test_df[feature_cols]
    y_test = test_df['pluie_demain_bin']
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_catboost_classifier(X_train, y_train, X_val, y_val):
    # Identify categorical features (object dtype) for CatBoost
    cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype == 'object']
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=100,
        random_seed=42,
        thread_count=4,
    )
    # Pass cat_features to fit via cat_features parameter
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, early_stopping_rounds=50)
    return model

def save_model(model, name):
    path = os.path.join(MODEL_DIR, f"{name}.cbm")
    model.save_model(path)
    print(f"Model saved to {path}")

def evaluate(model, X, y, split_name):
    preds = model.predict_proba(X)[:, 1]
    pred_labels = (preds > 0.5).astype(int)
    auc = roc_auc_score(y, preds)
    print(f"=== {split_name} ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y, pred_labels))
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{split_name}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(classification_report(y, pred_labels))
    print(f"Metrics saved to {metrics_path}")

def train_keras_classifier(X_train, y_train, X_val, y_val, X_test, epochs=30, batch_size=256):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # Encode categorical object columns to integer codes for Keras
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    for col in X_train_enc.select_dtypes(include='object').columns:
        X_train_enc[col], uniques = pd.factorize(X_train_enc[col])
        # Align validation and test using same mapping (unknown categories become -1)
        X_val_enc[col] = X_val_enc[col].map({v:k for k,v in enumerate(uniques)})
        X_val_enc[col] = X_val_enc[col].fillna(-1).astype(int)
        X_test_enc[col] = X_test_enc[col].map({v:k for k,v in enumerate(uniques)})
        X_test_enc[col] = X_test_enc[col].fillna(-1).astype(int)
    model = keras.Sequential([
        layers.Input(shape=(X_train_enc.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    model.fit(X_train_enc, y_train, validation_data=(X_val_enc, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
    return model, X_test_enc
# Removed duplicated Keras training block that was unreachable after return


def save_keras_model(model, name='nn_classifier.keras'):
    path = os.path.join(MODEL_DIR, name)
    model.save(path)
    print(f"Keras model saved to {path}")

def main():
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    cat_model = train_catboost_classifier(X_train, y_train, X_val, y_val)
    save_model(cat_model, "catboost_binary_rain")
    evaluate(cat_model, X_val, y_val, "validation")
    evaluate(cat_model, X_test, y_test, "test")
    # Train and save Keras model
    keras_model, X_test_enc = train_keras_classifier(X_train, y_train, X_val, y_val, X_test)
    save_keras_model(keras_model)
    # Optional: evaluate Keras model (using predict)
    preds = keras_model.predict(X_test_enc).ravel()
    pred_labels = (preds > 0.5).astype(int)
    auc = roc_auc_score(y_test, preds)
    print("=== Keras test ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, pred_labels))
    # Save Keras metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics_keras_test.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(classification_report(y_test, pred_labels))
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
