import os
import pandas as pd
import numpy as np

def build_features(input_path, output_path):
    print("=" * 60)
    print("  RUNNING FEATURE ENGINEERING (STAGE 3)")
    print("=" * 60)
    
    # 0. Load Dataset
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort chronologically to ensure correct shifts and rolling calculations
    df = df.sort_values(by=["ville", "date"]).reset_index(drop=True)
    
    # 1. Build Lag Features (grouped by city to prevent cross-city leakage)
    print("Creating lag features (lags: 1, 3, 7)...")
    df["pluie_lag1"] = df.groupby("ville")["pluie"].shift(1)
    df["pluie_lag3"] = df.groupby("ville")["pluie"].shift(3)
    df["pluie_lag7"] = df.groupby("ville")["pluie"].shift(7)
    df["temp_lag1"] = df.groupby("ville")["temp_mean"].shift(1)
    df["pression_lag1"] = df.groupby("ville")["pression_mean"].shift(1)
    
    # 2. Build Rolling Statistics (grouped by city)
    print("Creating rolling statistics (windows: 7j, 30j)...")
    df["rolling_pluie_7j"] = df.groupby("ville")["pluie"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["rolling_pluie_30j"] = df.groupby("ville")["pluie"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["rolling_temp_7j"] = df.groupby("ville")["temp_mean"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["rolling_humidite_7j"] = df.groupby("ville")["humidite_mean"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # 3. Create Derived Meteorological Indicators
    print("Creating derived indicators (delta_pression, stress_hydrique, ratio_ensoleillement)...")
    df["delta_pression"] = df["pression_max"] - df["pression_min"]
    df["stress_hydrique"] = df["evapotranspiration"] - df["precipitation"]
    df["ratio_ensoleillement"] = (df["ensoleillement_h"] / (df["duree_jour_h"] + 1e-5)).clip(0, 1)
    
    # 4. Handle NaNs in Lag Features
    # The first few rows of each city will have NaNs for lags. We fill these NaNs
    # using medians computed on the training set (<= 2022) to prevent data leakage!
    new_numerical_cols = [
        "pluie_lag1", "pluie_lag3", "pluie_lag7", "temp_lag1", "pression_lag1",
        "rolling_pluie_7j", "rolling_pluie_30j", "rolling_temp_7j", "rolling_humidite_7j",
        "delta_pression", "stress_hydrique", "ratio_ensoleillement"
    ]
    
    print("Imputing lag/rolling NaNs (using train split medians by city/month to avoid leakage)...")
    train_mask = df["date"].dt.year <= 2022
    train_df = df[train_mask]
    
    medians_by_group = train_df.groupby(["ville", "mois"])[new_numerical_cols].median()
    global_medians = train_df[new_numerical_cols].median()
    medians_by_group = medians_by_group.fillna(global_medians)
    
    df_indexed = df.set_index(["ville", "mois"])
    for col in new_numerical_cols:
        missing_before = df_indexed[col].isnull().sum()
        if missing_before > 0:
            df_indexed[col] = df_indexed[col].fillna(medians_by_group[col])
            missing_after = df_indexed[col].isnull().sum()
            print(f"  Column '{col}': imputed {missing_before - missing_after} lag NaNs (remaining: {missing_after})")
            
    df = df_indexed.reset_index()
    
    # 5. Seasonal Interactions
    print("Creating seasonal interaction features...")
    # Add dummy columns for saison
    seasons_ohe = pd.get_dummies(df["saison"], prefix="saison", dtype=int)
    df = pd.concat([df, seasons_ohe], axis=1)
    
    # Create interactions between season dummies and core lag features
    for s_col in seasons_ohe.columns:
        df[f"{s_col}_x_pluie_lag1"] = df[s_col] * df["pluie_lag1"]
        df[f"{s_col}_x_temp_lag1"] = df[s_col] * df["temp_lag1"]
        df[f"{s_col}_x_pression_lag1"] = df[s_col] * df["pression_lag1"]
        
    # Rearrange columns to put date and city first
    cols = ["date", "ville"] + [c for c in df.columns if c not in ["date", "ville"]]
    df = df[cols]
    
    # Save the feature engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    print(f"  Feature-rich dataset saved to: {output_path}")
    print(f"  Final rows count: {len(df):,}")
    print(f"  Final columns count: {df.shape[1]}")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/tunisie_meteo_clean.csv")
    output_path = os.path.join(base_dir, "data/tunisie_meteo_features.csv")
    
    build_features(input_path, output_path)
