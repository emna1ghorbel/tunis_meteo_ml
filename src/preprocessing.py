import os
import pandas as pd
import numpy as np
import sys

def preprocess_data(data_path, output_path):
    print("=" * 60)
    print("  RUNNING PREPROCESSING & DATA CLEANING (STAGE 2)")
    print("=" * 60)
    
    # 0. Load Dataset
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Fix target evaluation bug: NaNs in pluie_demain_mm should remain NaN in pluie_demain_bin
    df["pluie_demain_bin"] = np.where(df["pluie_demain_mm"].isna(), np.nan, (df["pluie_demain_mm"] > 1.0).astype(float))
    
    # Sort chronologically by city and date
    df = df.sort_values(by=["ville", "date"]).reset_index(drop=True)
    
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows:,}")
    
    # 1. Split Temporel definition (Train: 2009-2022)
    # We fit medians on train set only to prevent data leakage!
    train_mask = df["date"].dt.year <= 2022
    train_df = df[train_mask]
    print(f"Training set size (for fitting medians): {len(train_df):,} rows")
    
    # 2. Compute Medians by City and Month on Training set
    print("Computing imputation medians by city and month (from train set only)...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and index columns from imputation features list
    for col_to_remove in ["pluie_demain_mm", "pluie_demain_bin", "mois"]:
        if col_to_remove in numerical_cols:
            numerical_cols.remove(col_to_remove)
            
    medians_by_group = train_df.groupby(["ville", "mois"])[numerical_cols].median()
    global_medians = train_df[numerical_cols].median()
    
    # Fill any NaNs in the group medians with the global training medians
    medians_by_group = medians_by_group.fillna(global_medians)
    
    # 3. Impute Missing Values in the full dataset
    print("Imputing missing values using computed medians...")
    df_indexed = df.set_index(["ville", "mois"])
    
    for col in numerical_cols:
        missing_before = df_indexed[col].isnull().sum()
        if missing_before > 0:
            df_indexed[col] = df_indexed[col].fillna(medians_by_group[col])
            missing_after = df_indexed[col].isnull().sum()
            print(f"  Column '{col}': imputed {missing_before - missing_after} NaNs (remaining: {missing_after})")
            
    df = df_indexed.reset_index()
    
    # 4. Handle Missing Target values (pluie_demain_bin)
    target_missing = df["pluie_demain_bin"].isnull().sum()
    print(f"Missing values in target 'pluie_demain_bin': {target_missing}")
    if target_missing > 0:
        print("Dropping rows with missing target values...")
        df = df.dropna(subset=["pluie_demain_bin"])
        df["pluie_demain_bin"] = df["pluie_demain_bin"].astype(int)
        print(f"Row count after dropping missing target: {len(df):,}")
        
    # Also handle missing target values in pluie_demain_mm
    df["pluie_demain_mm"] = df["pluie_demain_mm"].fillna(0.0)
    
    # 5. One-Hot Encoding for 'ville'
    print("Applying One-Hot Encoding on 'ville'...")
    # Keep the original 'ville' column for grouping, and add one-hot encoded columns
    villes_ohe = pd.get_dummies(df["ville"], prefix="ville", dtype=int)
    df = pd.concat([df, villes_ohe], axis=1)
    print(f"Encoded city columns added: {', '.join(villes_ohe.columns)}")
    
    # Restore standard column ordering (date first, ville second, then others)
    cols = ["date", "ville"] + [c for c in df.columns if c not in ["date", "ville"]]
    df = df[cols]
    
    # 6. Save preprocessed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"  Clean dataset saved to: {output_path}")
    print(f"  Final rows count: {len(df):,}")
    print(f"  Final columns count: {df.shape[1]}")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data/tunisie_meteo_reelle_2009_2026.csv")
    output_path = os.path.join(base_dir, "data/tunisie_meteo_clean.csv")
    
    preprocess_data(data_path, output_path)
