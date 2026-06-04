import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def build_feature_learning(input_path, output_path, models_dir):
    print("=" * 60)
    print("  RUNNING AUTOMATIC FEATURE LEARNING (STAGE 4)")
    print("=" * 60)
    
    # 0. Load Dataset
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["ville", "date"]).reset_index(drop=True)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Identify splits
    train_mask = df["date"].dt.year <= 2022
    val_mask = df["date"].dt.year == 2023
    test_mask = df["date"].dt.year >= 2024
    
    # 2. Define Weather Features for Scaling and Dimensionality Reduction
    core_weather_cols = [
        "temp_max", "temp_min", "temp_mean", "precipitation", "pluie", "neige_cm",
        "heures_pluie", "vent_max", "rafales_max", "vent_direction", "rayonnement",
        "evapotranspiration", "ensoleillement_h", "duree_jour_h", "humidite_max",
        "humidite_min", "humidite_mean", "rosee_max", "rosee_min", "ressenti_max",
        "ressenti_min", "pression_max", "pression_min", "pression_mean", "nuages_pct",
        "temp_sol", "humidite_sol", "amplitude_temp", "stress_hydrique", "ratio_ensoleillement"
    ]
    
    # Ensure they are present
    core_weather_cols = [c for c in core_weather_cols if c in df.columns]
    
    # 3. Fit StandardScaler on Training Set only (to avoid leakage)
    print("Standardizing features (StandardScaler fit on train only)...")
    scaler = StandardScaler()
    
    # We copy df to avoid SettingWithCopyWarning
    df_scaled = df.copy()
    scaler.fit(df_scaled.loc[train_mask, core_weather_cols])
    df_scaled[core_weather_cols] = scaler.transform(df_scaled[core_weather_cols])
    
    # 4. PCA on Temperature related features
    print("Fitting PCA (3 components) on temperature features...")
    temp_cols = ["temp_max", "temp_min", "temp_mean", "amplitude_temp", "rosee_max", "rosee_min"]
    temp_cols = [c for c in temp_cols if c in core_weather_cols]
    
    pca = PCA(n_components=3, random_state=42)
    pca.fit(df_scaled.loc[train_mask, temp_cols])
    
    pca_features = pca.transform(df_scaled[temp_cols])
    for i in range(3):
        df[f"pca_{i+1}"] = pca_features[:, i]
    print(f"  PCA components added. Explained variance: {pca.explained_variance_ratio_}")
    
    # 5. Train Keras Autoencoder on core weather features
    print("Building and training Keras Autoencoder...")
    input_dim = len(core_weather_cols)
    
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(16, name='bottleneck', activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    X_train_ae = df_scaled.loc[train_mask, core_weather_cols].values
    X_val_ae = df_scaled.loc[val_mask, core_weather_cols].values
    
    early_stop_ae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training autoencoder...")
    autoencoder.fit(
        X_train_ae, X_train_ae,
        validation_data=(X_val_ae, X_val_ae),
        epochs=30,
        batch_size=128,
        callbacks=[early_stop_ae],
        verbose=0
    )
    
    # Extract Autoencoder Bottleneck Features
    print("Extracting Autoencoder representation features...")
    ae_features = encoder.predict(df_scaled[core_weather_cols].values)
    for i in range(16):
        df[f"ae_{i+1}"] = ae_features[:, i]
        
    # Save autoencoder model weights
    autoencoder.save(os.path.join(models_dir, "autoencoder_model.h5"))
    print("  Autoencoder features added.")
    
    # 6. Train Keras LSTM on 14-day sliding windows
    print("Preparing 14-day sliding windows for LSTM...")
    time_steps = 14
    X_seq = []
    y_seq = []
    
    # Standardized weather features array
    weather_data_scaled = df_scaled[core_weather_cols].values
    targets = df["pluie_demain_bin"].values
    villes = df["ville"].values
    
    # Generate windows with pre-padding of zeros for boundary safety
    for idx in range(len(df)):
        current_ville = villes[idx]
        # Look backwards up to 14 days, but stay within the same city
        start_idx = max(0, idx - time_steps + 1)
        # Check if the backward search crosses over to another city
        # Since we sorted by city first, if villes[start_idx] != current_ville, we find where the city starts
        city_start_idx = idx
        while city_start_idx > start_idx:
            if villes[city_start_idx - 1] != current_ville:
                break
            city_start_idx -= 1
            
        # Slice the sequence
        seq_slice = weather_data_scaled[city_start_idx : idx + 1]
        
        # Pre-pad with zeros if sequence length < 14
        if len(seq_slice) < time_steps:
            pad_len = time_steps - len(seq_slice)
            pad_block = np.zeros((pad_len, input_dim))
            seq_slice = np.vstack([pad_block, seq_slice])
            
        X_seq.append(seq_slice)
        y_seq.append(targets[idx])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Prepare sequence splits
    X_train_lstm = X_seq[train_mask]
    y_train_lstm = y_seq[train_mask]
    X_val_lstm = X_seq[val_mask]
    y_val_lstm = y_seq[val_mask]
    
    print(f"LSTM Train sequences shape: {X_train_lstm.shape}")
    
    # Build LSTM Model
    inputs_lstm = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True, dropout=0.3)(inputs_lstm)
    lstm_out = LSTM(64, return_sequences=False, dropout=0.3)(x)
    outputs_lstm = Dense(1, activation='sigmoid')(lstm_out)
    
    lstm_model = Model(inputs=inputs_lstm, outputs=outputs_lstm)
    # Define an embedding extractor model that goes up to the second LSTM layer output
    lstm_extractor = Model(inputs=inputs_lstm, outputs=lstm_out)
    
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop_lstm = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training LSTM for supervised representation learning...")
    # Using small class weight for imbalance handling during representation learning
    class_weight_0 = 1.0
    class_weight_1 = len(y_train_lstm[y_train_lstm == 0]) / len(y_train_lstm[y_train_lstm == 1])
    class_weights = {0: class_weight_0, 1: class_weight_1}
    
    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=15,
        batch_size=128,
        class_weight=class_weights,
        callbacks=[early_stop_lstm],
        verbose=0
    )
    
    # Extract LSTM Temporal Embeddings
    print("Extracting LSTM temporal embedding features...")
    lstm_features = lstm_extractor.predict(X_seq)
    for i in range(64):
        df[f"lstm_{i+1}"] = lstm_features[:, i]
        
    # Save LSTM models
    lstm_model.save(os.path.join(models_dir, "lstm_supervised_model.h5"))
    print("  LSTM temporal embedding features added.")
    
    # 7. Save Consolidated Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print("  FEATURE LEARNING COMPLETED")
    print("=" * 60)
    print(f"  Final consolidated dataset saved to: {output_path}")
    print(f"  Final rows count: {len(df):,}")
    print(f"  Final columns count: {df.shape[1]}")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/tunisie_meteo_features.csv")
    output_path = os.path.join(base_dir, "data/tunisie_meteo_final.csv")
    models_dir = os.path.join(base_dir, "models")
    
    build_feature_learning(input_path, output_path, models_dir)
