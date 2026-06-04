import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for high-quality, premium visual aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150
})

# Color palette definition (Harmonious Teal/Coral/Blue/Gray)
SEASONS_PALETTE = {
    "Hiver": "#1f77b4",       # Cool blue
    "Printemps": "#2ca02c",   # Fresh green
    "Été": "#ff7f0e",        # Sunny orange
    "Automne": "#9467bd"      # Warm purple
}
CLASS_PALETTE = ["#8da0cb", "#fc8d62"] # Sleek blue-gray and coral

def run_eda(data_path, results_dir):
    print("=" * 60)
    print("  RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 0. Load Dataset
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Class Imbalance Analysis on target 'pluie_demain_bin'
    print("\n1. Analyzing class imbalance...")
    target_col = "pluie_demain_bin"
    vc = df[target_col].value_counts(dropna=False)
    vc_pct = df[target_col].value_counts(normalize=True, dropna=False) * 100
    
    print("Target distribution:")
    for idx, val in vc.items():
        print(f"  Class {idx}: {val:5,d} ({vc_pct[idx]:.2f}%)")
        
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x=target_col, palette=CLASS_PALETTE, ax=ax)
    ax.set_title("Distribution de la cible 'pluie_demain_bin'\n(Déséquilibre de classe)")
    ax.set_xlabel("Il pleuvra demain (0 = Non, 1 = Oui)")
    ax.set_ylabel("Nombre de jours")
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:,.0f}\n({height/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height + 100),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "01_class_imbalance.png"))
    plt.close(fig)
    print(f"Saved: {os.path.join(results_dir, '01_class_imbalance.png')}")
    
    # 2. Descriptive statistics by City
    print("\n2. Generating descriptive statistics by city...")
    desc_cols = ["temp_max", "temp_min", "temp_mean", "precipitation", "humidite_mean", "vent_max", "pression_mean"]
    desc_stats = df.groupby("ville")[desc_cols].describe()
    
    # Save descriptive stats report
    desc_stats.to_csv(os.path.join(results_dir, "descriptive_stats_by_city.csv"))
    print(f"Saved descriptive stats: {os.path.join(results_dir, 'descriptive_stats_by_city.csv')}")
    
    # Print clean summary stats
    print("\nSummary statistics (Mean values by City):")
    mean_stats = df.groupby("ville")[desc_cols].mean()
    print(mean_stats.round(2).to_string())
    
    # 3. Missing values analysis
    print("\n3. Analyzing missing values...")
    na_counts = df.isnull().sum()
    na_pcts = (df.isnull().sum() / len(df)) * 100
    na_df = pd.DataFrame({"Missing Count": na_counts, "Percentage (%)": na_pcts})
    na_df = na_df.sort_values(by="Missing Count", ascending=False)
    
    # Save missing values report
    na_df.to_csv(os.path.join(results_dir, "missing_values_report.csv"))
    print(f"Saved missing values report: {os.path.join(results_dir, 'missing_values_report.csv')}")
    
    print("\nTop 15 columns with missing values:")
    print(na_df.head(15).to_string())
    
    # 4. Distributions of temp_mean, precipitation, humidite_mean by City and Season
    print("\n4. Plotting distributions by city and season...")
    features_to_plot = ["temp_mean", "precipitation", "humidite_mean"]
    villes = df["ville"].unique()
    
    # Grid of 4 rows (cities) x 3 columns (features)
    fig, axes = plt.subplots(nrows=len(villes), ncols=len(features_to_plot), figsize=(16, 14), sharex="col")
    
    for row_idx, city in enumerate(villes):
        city_df = df[df["ville"] == city]
        for col_idx, col_name in enumerate(features_to_plot):
            ax = axes[row_idx, col_idx]
            
            # For precipitation, use log scale or histplot since it's highly skewed
            if col_name == "precipitation":
                sns.histplot(data=city_df, x=col_name, hue="saison", palette=SEASONS_PALETTE,
                             element="step", stat="density", common_norm=False, bins=30, ax=ax, alpha=0.4)
                ax.set_xlim(-1, 25) # zoom in on common precipitation ranges
            else:
                sns.kdeplot(data=city_df, x=col_name, hue="saison", palette=SEASONS_PALETTE,
                            fill=True, common_norm=False, alpha=0.3, ax=ax)
                
            # Add city name on the left y-label
            if col_idx == 0:
                ax.set_ylabel(f"{city}\nDensity")
            else:
                ax.set_ylabel("")
                
            # Add titles only for the first row
            if row_idx == 0:
                ax.set_title(f"Distribution of {col_name}")
                
            # Clean legends
            if row_idx != 0 or col_idx != 2:
                if ax.get_legend():
                    ax.get_legend().remove()
    
    fig.suptitle("Distributions de temp_mean, precipitation, et humidite_mean par ville et saison", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "02_distributions_by_city_season.png"))
    plt.close(fig)
    print(f"Saved: {os.path.join(results_dir, '02_distributions_by_city_season.png')}")
    
    # 5. Correlation matrix heatmap
    print("\n5. Plotting correlation heatmap...")
    corr_cols = [
        "temp_max", "temp_min", "temp_mean", "precipitation", "pluie", "heures_pluie",
        "vent_max", "rafales_max", "vent_direction", "rayonnement", "evapotranspiration",
        "ensoleillement_h", "humidite_max", "humidite_min", "humidite_mean", "rosee_max",
        "rosee_min", "ressenti_max", "ressenti_min", "pression_max", "pression_min",
        "pression_mean", "nuages_pct", "temp_sol", "humidite_sol"
    ]
    # Filter available columns
    corr_cols = [c for c in corr_cols if c in df.columns]
    
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    # We will use mask to only show lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", fmt=".2f",
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title("Matrice de Corrélation des Caractéristiques Météorologiques (Triangle Inférieur)")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "03_correlation_matrix.png"))
    plt.close(fig)
    print(f"Saved: {os.path.join(results_dir, '03_correlation_matrix.png')}")
    
    # 6. Seasonality visualization (Monthly boxplots x Ville for temp_mean)
    print("\n6. Plotting seasonality boxplots...")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    for idx, city in enumerate(villes):
        ax = axes_flat[idx]
        city_df = df[df["ville"] == city]
        sns.boxplot(data=city_df, x="mois", y="temp_mean", hue="saison",
                    palette=SEASONS_PALETTE, dodge=False, ax=ax)
        ax.set_title(f"Saisonnalité de temp_mean - {city}")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Température Moyenne (°C)" if idx % 2 == 0 else "")
        # Remove legend except for the last subplot
        if idx != 3:
            ax.get_legend().remove()
        else:
            ax.legend(title="Saison", bbox_to_anchor=(1.05, 1), loc='upper left')
            
    fig.suptitle("Variation Mensuelle et Saisonnière de la Température Moyenne par Ville", y=0.96)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "04_seasonality_boxplots.png"))
    plt.close(fig)
    print(f"Saved: {os.path.join(results_dir, '04_seasonality_boxplots.png')}")
    
    print("\n" + "=" * 60)
    print("  EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Weather Forecasting EDA")
    parser.add_argument("--data", type=str, default="data/tunisie_meteo_reelle_2009_2026.csv", help="Path to input dataset")
    parser.add_argument("--results", type=str, default="results", help="Directory to save output plots and reports")
    args = parser.parse_args()
    
    # Resolve absolute or project paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data
    results_dir = os.path.join(base_dir, args.results) if not os.path.isabs(args.results) else args.results
    
    run_eda(data_path, results_dir)
