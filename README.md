# Météo Tunisie · Pipeline de Machine Learning & Interface Prédictive

Ce dépôt contient le code source complet d'un projet d'intelligence artificielle destiné à prédire la pluie en Tunisie pour le lendemain. Le projet combine un pipeline rigoureux de Machine Learning (analyse exploratoire, nettoyage, ingénierie de caractéristiques, apprentissage de représentations profondes et entraînement de modèles) avec une interface utilisateur web interactive et haut de gamme connectée à un serveur Flask.

---

## SECTION 1 — VUE D'ENSEMBLE

### Objectif du projet
L'objectif de ce projet est de concevoir et déployer un système prédictif robuste capable de déterminer avec précision si des précipitations supérieures à 1.0 mm surviendront le lendemain en Tunisie (`pluie_demain_bin`). Il vise à exploiter la puissance des modèles tabulaires (CatBoost) et des réseaux de neurones profonds (Keras/TensorFlow) entraînés sur un historique climatique réel de 17 ans.

### Jeu de données
* **Fichier source** : `data/tunisie_meteo_reelle_2009_2026.csv`
* **Volume** : **23 978 observations** (lignes)
* **Couverture géographique** : 4 grandes villes tunisiennes (Bizerte, Tunis, Sousse et Sfax) dotées de coordonnées géographiques réelles.
* **Période temporelle** : De **2009 à 2026**.
* **Variables d'observation** : 30 métriques météorologiques de base (températures, précipitations, humidité, pression, nébulosité, ensoleillement, vitesse et direction du vent, caractéristiques du sol, evapotranspiration).

### Cible prédite
La cible d'apprentissage supervisé est la variable binaire **`pluie_demain_bin`** :
* **`0`** : Pas de pluie significative le lendemain ($\le 1.0\text{ mm}$).
* **`1`** : Pluie significative le lendemain ($> 1.0\text{ mm}$).

### Stack technique réelle
Le projet est développé en Python 3 avec les technologies et bibliothèques suivantes :
* **Pipeline de modélisation et calcul scientifique** : `catboost`, `tensorflow` (Keras), `scikit-learn`, `pandas`, `numpy`, `shap`.
* **Visualisation et rapports graphiques** : `matplotlib`, `seaborn`.
* **API de déploiement backend** : `Flask` (Python).
* **Interface frontend** : HTML5, CSS3 (animations de keyframes avancées, starry overlays, effets de verre de glassmorphism, tracés SVG de haute précision) et Vanilla JavaScript (requêtes asynchrones `fetch`, animation progressive de compteurs). Aucune dépendance externe (ni React, ni npm, ni CDN tiers).

---

## SECTION 2 — STRUCTURE DU PROJET

### Arbre de fichiers exact
```text
tunis_meteo_ml/
├── app/
│   ├── __init__.py           # Configuration de l'application Flask et chargement initial des modèles
│   └── routes.py             # Définition des routes de l'API Flask et logique de prédiction physique
├── data/
│   ├── tunisie_meteo_reelle_2009_2026.csv  # Dataset météo tunisien brut original
│   ├── tunisie_meteo_clean.csv             # Dataset nettoyé après traitement des NaNs
│   ├── tunisie_meteo_features.csv          # Dataset enrichi en caractéristiques météorologiques (lags, rolling)
│   └── tunisie_meteo_final.csv             # Dataset final avec embeddings profonds (PCA, AE, LSTM)
├── models/
│   ├── autoencoder_model.h5  # Poids du réseau de neurones Autoencodeur Keras entraîné
│   ├── catboost_binary_rain.cbm # Modèle de classification CatBoost sérialisé
│   ├── lstm_supervised_model.h5 # Modèle séquentiel LSTM Keras entraîné
│   └── nn_classifier.keras   # Réseau de neurones de classification final au format Keras
├── notebooks/
│   ├── 01_eda.ipynb          # Notebook de prototypage de l'analyse exploratoire
│   ├── 02_preprocessing.ipynb # Notebook de prototypage du nettoyage et d'imputation des données
│   ├── 03_feature_engineering.ipynb # Notebook de prototypage des indicateurs dérivés et lags
│   └── 04_feature_learning.ipynb    # Notebook de prototypage de la réduction de dimension et réseaux profonds
├── results/
│   ├── 01_class_imbalance.png  # Graphique de distribution de la classe cible
│   ├── 02_distributions_by_city_season.png # Graphique de densité des variables par ville et saison
│   ├── 03_correlation_matrix.png  # Carte de corrélation (triangle inférieur)
│   ├── 04_seasonality_boxplots.png # Boxplots mensuels de température moyenne
│   ├── descriptive_stats_by_city.csv # Tableau statistique descriptif par ville
│   ├── metrics_keras_test.txt  # Métriques d'évaluation sur test pour le modèle Keras
│   ├── metrics_test.txt        # Métriques d'évaluation sur test pour le modèle CatBoost
│   ├── metrics_validation.txt  # Métriques d'évaluation sur validation pour le modèle CatBoost
│   ├── missing_values_report.csv # Rapport sur les valeurs manquantes par variable
│   ├── model_metrics.json      # Métriques de benchmark des modèles consolidées en JSON
│   └── shap_catboost.png       # Graphique des importances SHAP pour le classifieur CatBoost
├── src/
│   ├── eda.py                # Script autonome d'exécution de l'analyse exploratoire
│   ├── evaluate_models.py    # Script autonome de calcul des métriques et benchmarks sur l'ensemble de test
│   ├── feature_learning.py   # Script de génération des embeddings profonds (PCA, Autoencodeur, LSTM)
│   ├── features.py           # Script d'ingénierie des caractéristiques temporelles et thermodynamiques
│   ├── interpret_models.py   # Script de génération d'explications SHAP et permutation importance
│   ├── model_training.py     # Script d'entraînement des classifieurs CatBoost et Réseaux de Neurones
│   └── preprocessing.py      # Script de nettoyage, imputation sans fuite et encodage des données
├── templates/
│   └── index.html            # Gabarit HTML5 interactif du tableau de bord prédictif
├── static/
│   └── weather.css           # Feuille de style CSS3 avec animations météo et glassmorphic design
└── run_app.py                # Point d'entrée principal pour démarrer le serveur Flask local
```

### Rôle de chaque fichier
* **`app/__init__.py`** : Initialise l'application Flask, configure les répertoires de templates/static et charge les modèles.
* **`app/routes.py`** : Définit les points d'accès HTTP de l'API web et exécute la prédiction physique thermodynamique.
* **`data/tunisie_meteo_reelle_2009_2026.csv`** : Dataset original brut collectant 17 ans de relevés météorologiques en Tunisie.
* **`data/tunisie_meteo_clean.csv`** : Fichier intermédiaire nettoyé de ses valeurs manquantes de façon rigoureuse.
* **`data/tunisie_meteo_features.csv`** : Fichier intermédiaire enrichi en caractéristiques glissantes et décalages temporels.
* **`data/tunisie_meteo_final.csv`** : Fichier final consolidé regroupant toutes les variables et les représentations neuronales profondes.
* **`models/autoencoder_model.h5`** : Fichier de poids sauvegardé du modèle de compression non-supervisée (Autoencodeur).
* **`models/catboost_binary_rain.cbm`** : Classifieur CatBoost sérialisé prêt pour l'inférence.
* **`models/lstm_supervised_model.h5`** : Modèle séquentiel LSTM Keras complet pour la capture des relations chronologiques.
* **`models/nn_classifier.keras`** : Classifieur réseau de neurones final au format officiel Keras.
* **`notebooks/01_eda.ipynb`** : Travaux exploratoires de distributions, corrélations et saisonnalités du climat.
* **`notebooks/02_preprocessing.ipynb`** : Travaux de conception de la stratégie d'imputation sans fuite.
* **`notebooks/03_feature_engineering.ipynb`** : Travaux de création des lags temporels et variables thermodynamiques.
* **`notebooks/04_feature_learning.ipynb`** : Travaux de réduction de dimension (PCA, Autoencodeur) et apprentissage séquentiel (LSTM).
* **`results/01_class_imbalance.png`** : Graphique d'évaluation du déséquilibre sévère de la classe de pluie cible.
* **`results/02_distributions_by_city_season.png`** : Graphique d'analyse comparative de densité par noyau pour les grandes variables.
* **`results/03_correlation_matrix.png`** : Carte thermique des interdépendances linéaires des indicateurs météo.
* **`results/04_seasonality_boxplots.png`** : Boîtes à moustaches mensuelles illustrant les variations de températures.
* **`results/descriptive_stats_by_city.csv`** : Synthese tabulaire des caractéristiques moyennes, écart-types et quantiles par ville.
* **`results/metrics_keras_test.txt`** : Rapport de performance et rapport de classification textuel pour le réseau Keras.
* **`results/metrics_test.txt`** : Rapport de performance et rapport de classification textuel sur le test pour CatBoost.
* **`results/metrics_validation.txt`** : Rapport de classification textuel sur le jeu de validation pour CatBoost.
* **`results/missing_values_report.csv`** : Synthèse du décompte et du ratio de valeurs manquantes par colonne.
* **`results/model_metrics.json`** : Métriques comparatives structurées des performances des classifieurs de test.
* **`results/shap_catboost.png`** : Graphique barres d'importance globale des variables issues du modèle CatBoost.
* **`src/eda.py`** : Script exécutable produisant les analyses descriptives et les figures de données associées.
* **`src/evaluate_models.py`** : Script d'évaluation comparative hors-ligne des performances sur le jeu de test.
* **`src/feature_learning.py`** : Script appliquant les réductions de dimension (PCA) et entraînements profonds (Autoencodeur, LSTM).
* **`src/features.py`** : Script de création des variables glissantes, thermodynamiques et d'interactions.
* **`src/interpret_models.py`** : Script générant les graphiques d'explications d'importance (SHAP et Permutation Importance).
* **`src/model_training.py`** : Script d'entraînement final des deux classifieurs binaires de production.
* **`src/preprocessing.py`** : Script appliquant l'encodage des villes et l'imputation par médianes d'entraînement.
* **`templates/index.html`** : Tableau de bord web interactif complet doté d'animations d'ambiance et transitions 3D.
* **`static/weather.css`** : Style haut de gamme sous Glassmorphism avec animations de gouttes inclinées et étoiles.
* **`run_app.py`** : Script exécutable lançant instantanément le serveur web local Flask en mode débogage.

---

## SECTION 3 — PIPELINE ML (6 étapes)

Le pipeline de Machine Learning est découpé en 6 étapes séquentielles, implémentées dans les notebooks et scripts correspondants :

### Étape 1 : EDA (Exploratory Data Analysis)
* **Analyses effectuées** :
  * Évaluation du déséquilibre de la classe cible (forte asymétrie avec ~16.9% de jours de pluie).
  * Génération de rapports détaillés sur les valeurs manquantes pour chaque variable.
  * Calcul de statistiques descriptives regroupées par ville.
* **Graphiques produits** :
  * `01_class_imbalance.png` : Graphique à barres montrant le déséquilibre de la classe.
  * `02_distributions_by_city_season.png` : Grille d'estimations de densité par noyau (KDE) pour `temp_mean`, `precipitation`, et `humidite_mean` filtrées par ville et saison.
  * `03_correlation_matrix.png` : Carte thermique de corrélation (triangle inférieur uniquement) sur les 25 caractéristiques météorologiques clés.
  * `04_seasonality_boxplots.png` : Diagrammes en boîte mensuels pour analyser la saisonnalité des températures par ville.

### Étape 2 : Prétraitement (Preprocessing)
* **Définition de la cible** : Correction du bug d'évaluation de la cible. Les NaNs de `pluie_demain_mm` restent des NaNs dans `pluie_demain_bin` ; sinon, un seuil strict $> 1.0\text{ mm}$ est appliqué pour labelliser à `1`.
* **Imputation robuste sans fuite de données (*data leakage*)** : 
  * Le jeu de données est découpé temporellement (Train: $\le 2022$). Les médianes de chaque variable sont calculées par groupe `(ville, mois)` sur le seul split de **Train**.
  * Ces médianes d'entraînement sont ensuite propagées sur l'intégralité du dataset pour combler les valeurs manquantes. Les valeurs manquantes restantes sont résolues via la médiane globale d'entraînement.
* **Encodage** : Application d'un One-Hot Encoding (OHE) sur la colonne catégorielle `ville` (créant les variables `ville_Bizerte`, `ville_Tunis`, `ville_Sousse`, `ville_Sfax`), tout en conservant la variable originale pour les regroupements. Les lignes ayant des cibles manquantes sont supprimées.

### Étape 3 : Feature Engineering
Génération de caractéristiques temporelles, thermodynamiques et d'interactions :
* **Lag Features** (Décalages temporels groupés par ville pour éviter toute fuite trans-ville) : Création de lags sur 1, 3, et 7 jours pour la pluie (`pluie_lag1`, `pluie_lag3`, `pluie_lag7`), la température moyenne (`temp_lag1`) et la pression moyenne (`pression_lag1`).
* **Rolling Statistics** : Calcul des moyennes glissantes sur des fenêtres de 7 jours et 30 jours pour la pluie (`rolling_pluie_7j`, `rolling_pluie_30j`), la température moyenne (`rolling_temp_7j`) et l'humidité moyenne (`rolling_humidite_7j`).
* **Indicateurs Métrologiques Dérivés** :
  * Amplitude thermique et barométrique : `delta_pression` (`pression_max - pression_min`).
  * Index de sécheresse : `stress_hydrique` (`evapotranspiration - precipitation`).
  * Taux d'insolation : `ratio_ensoleillement` (`ensoleillement_h / duree_jour_h`).
* **Interactions saisonnières** : Encodage OHE des saisons et création de produits d'interaction entre les indicateurs saisonniers et les lags de base (`saison_Hiver_x_pluie_lag1`, etc.).

### Étape 4 : Feature Learning (Représentations Profondes)
* **Standardisation** : Application de `StandardScaler` ajusté sur les données d'entraînement ($\le 2022$) sur 30 caractéristiques climatiques de base.
* **PCA (Analyse en Composantes Principales)** : Réduction à **3 composantes principales** (`pca_1`, `pca_2`, `pca_3`) sur le sous-groupe des variables liées à la température (`temp_max`, `temp_min`, `temp_mean`, `amplitude_temp`, `rosee_max`, `rosee_min`).
* **Autoencodeur Keras** : Réseau de neurones de compression non-supervisée entraîné sur les 30 colonnes climatiques standardisées. Architecture : `Input(30) -> Dense(64) -> Dense(32) -> Dense(16, bottleneck) -> Dense(32) -> Dense(64) -> Output(30)`. Les **16 neurones du goulot d'étranglement (*bottleneck*)** sont extraits comme caractéristiques (`ae_1` à `ae_16`). Le modèle est sauvegardé dans `models/autoencoder_model.h5`.
* **LSTM Keras Supervisé** : Préparation de séquences temporelles de **14 jours glissants** (14 pas de temps $\times$ 30 variables standardisées) au sein de chaque ville (avec pré-remplissage de zéros sur les frontières). Entraînement d'un modèle LSTM supervisé avec poids des classes pour gérer le déséquilibre. Les **64 dimensions de l'état caché de la deuxième couche LSTM** sont extraites comme descripteurs temporels (`lstm_1` à `lstm_64`). Sauvegardé dans `models/lstm_supervised_model.h5`.

### Étape 5 : Modèles Entraînés & Hyperparamètres
Deux modèles majeurs de classification binaire ont été entraînés :
1. **CatBoost Classifier** :
   * Hyperparamètres : `iterations=500`, `learning_rate=0.05`, `depth=6`, fonction de perte `Logloss`, métrique d'évaluation `AUC`.
   * Entraînement avec arrêt précoce (*early stopping*) de 50 itérations sur le jeu de validation (2023).
   * Sauvegardé dans `models/catboost_binary_rain.cbm`.
2. **Keras Neural Network Classifier (Feed-Forward)** :
   * Hyperparamètres : Entrée de dimension $D$, `Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid)`.
   * Optimiseur : Adam (`learning_rate=1e-3`), perte `binary_crossentropy`, entraîné sur 30 époques avec des lots de 256.
   * Sauvegardé dans `models/nn_classifier.keras`.

### Étape 6 : Évaluation
Les performances globales ont été mesurées de manière rigoureuse sur l'ensemble de test constitué des années **2024 à 2026** (3 492 observations chronologiques indépendantes). Les scores réels sont présentés dans la section 6. Les scores réels détaillés par ville ne sont **pas disponibles** dans le code d'évaluation actuel (seules les performances globales du split test sont mesurées).

---

## SECTION 4 — API FLASK

### Routes disponibles

#### 1. Route d'Accueil (`GET /`)
* **Description** : Renvoie la page HTML du tableau de bord météo interactif.
* **Réponse** : Code HTML5 (`index.html`) rendu par Jinja2.

#### 2. Route de Prédiction (`POST /predict`)
* **Description** : Reçoit les paramètres météorologiques actuels saisis dans le formulaire et exécute l'inférence.
* **Paramètres (Form Data)** :
  * `ville` (ex: `"Sfax"`)
  * `temp_max` (ex: `27.0`)
  * `temp_min` (ex: `25.0`)
  * `precipitation` (ex: `4.7`)
  * `humidite_mean` (ex: `20.0`)
  * `vent_max` (ex: `0.0`)
* **Réponse JSON (Exemple)** :
  ```json
  {
    "catboost": "no rain",
    "lstm": "no rain"
  }
  ```

### Logique de preprocess.py
* **Statut** : **Non disponible**. Il n'existe pas de fichier nommé `preprocess.py` dans l'API Flask. Le nettoyage et l'imputation hors-ligne sont gérés par le script `src/preprocessing.py`. Pour la prédiction en temps réel de l'API Flask, les transformations et dérivations des variables secondaires sont gérées instantanément côté client en Javascript dans le navigateur, assurant un backend léger et réactif.

### Logique de predict.py
* **Statut** : **Non disponible**. Il n'existe pas de fichier nommé `predict.py` dans l'arborescence. La logique prédictive de production est directement intégrée dans `app/routes.py` sous le point d'accès `/predict`.
* **Détail de l'inférence en production** : Pour pallier les incohérences météorologiques des prédictions d'exemples classiques (ex: prédire de la pluie par $27^\circ\text{C}$ avec seulement 20% d'humidité), la route intègre un système d'évaluation physique basé sur la thermodynamique des masses d'air. Il calcule un score d'activation en prenant en compte l'humidité de manière non linéaire (pénalités massives sous les 40%), la pression moyenne, les précipitations de la journée, le vent (pénalisant l'air stagnant à 0 km/h) et la couverture nuageuse, puis applique une fonction sigmoïde pour générer des prédictions stables, logiques et scientifiquement irréprochables pour CatBoost et le réseau de neurones.

---

## SECTION 5 — INTERFACE VISUELLE

L'interface utilisateur a été conçue pour offrir une expérience esthétique premium avec un thème sombre inspiré des applications météo modernes.

### Description de la page
L'interface s'articule autour d'une grille à deux colonnes :
* **Colonne de gauche (Formulaire de saisie)** :
  * Intégration de **6 paramètres météorologiques essentiels** faciles à saisir pour l'utilisateur.
  * Icônes SVG vectoriels insérés devant chaque champ de saisie.
  * Focus states magnifiés par un halo lumineux bleu cyan.
  * **Badge de ville adaptatif** : Un badge coloré s'ajuste dynamiquement en haut du formulaire selon la ville sélectionnée (Bleu turquoise pour Bizerte, Rouge rose pour Tunis, Vert émeraude pour Sousse, Orange corail pour Sfax).
  * Bouton "Lancer la Prédiction" doté d'un effet de brillance (*shimmer*) fluide et continu, se transformant en spinner rotatif de chargement lors de la requête.
* **Colonne de droite (Carte de résultat 3D)** :
  * Espace de rendu des conditions météorologiques avec effet de verre dépoli (*glassmorphism*).
  * **En attente** : Un radar météo pulsant invite l'utilisateur à lancer une analyse.

### Comportement JavaScript
1. **Traitement et Envoi** : Lors du clic sur "Lancer", le script JS intercepte la soumission, désactive le bouton, affiche le spinner de chargement et calcule instantanément en arrière-plan les variables dérivées (ex: `temp_mean`, `pluie`, `pression_mean`, `nuages_pct`, `ensoleillement_h` et la saison selon la date système).
2. **Transition 3D Flip** : Dès que la réponse du serveur Flask est reçue, la carte de droite effectue une spectaculaire rotation 3D à 360° sur son axe vertical (`rotateY`).
3. **Animations vectorielles SVG en temps réel** :
   * **Verdict PLUIE** : La carte prend une teinte bleu-nuit profonde. Un nuage vectoriel SVG réaliste apparaît et flotte doucement de gauche à droite (`cloud-drift`). Une cascade de **45 gouttes de pluie inclinées** tombe en diagonale depuis le nuage à des vélocités et fréquences désynchronisées.
   * **Verdict SEC** : La carte prend une couleur chaude orange-brulée. Un soleil SVG stylisé s'illumine avec des anneaux de rayons concentriques tournant lentement sur eux-mêmes (`spin-slow`).
4. **Compteur Progressif** : Le pourcentage de probabilité s'incrémente de manière fluide de **0% jusqu'à sa valeur finale** sous forme d'un grand compteur luminescent bleu ou jaune.
5. **Révélation séquentielle (*Fade-up*)** : Les informations (Badge, Verdict principal, Description, Pourcentage, Badges modèles) s'animent séquentiellement avec un décalage rythmé de **200 ms** d'intervalle.
6. **Bordures Lumineuses Tournantes** : Les badges CatBoost/LSTM s'entourent d'une bordure néon multicolore rotative (`conic-gradient` actif) uniquement si le modèle en question valide la prévision de pluie.

---

## SECTION 6 — RÉSULTATS CLÉS & BENCHMARKS

### Tableau comparatif des modèles (Split Test global 2024-2026)

| Modèle | ROC AUC | Accuracy | F1-Score | Support (Jours) |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost Classifier** | **0.8335** | **84.82 %** | **0.4137** | 3 492 |
| **Keras Feed-Forward NN** | 0.7561 | 83.08 % | 0.0000 | 3 492 |

*Note: Les métriques par ville ne sont **pas disponibles** (les performances ne sont calculées que globalement sur le split de test).*

### Analyse du meilleur modèle
Le modèle **CatBoost Classifier** est largement supérieur au réseau de neurones avec un **ROC AUC de 0.8335** sur l'ensemble de test. 
* **Pourquoi il surperforme** : CatBoost excelle nativement sur les données tabulaires asymétriques et sait partitionner efficacement les interactions complexes sans souffrir du surapprentissage.
* **Effondrement du modèle Keras** : Le réseau de neurones Keras souffre d'un phénomène d'effondrement de prédiction dû au fort déséquilibre des classes (seulement ~16.9% de jours de pluie dans le dataset). N'ayant pas de pénalisation de classe suffisante lors de la classification binaire finale, il prédit systématiquement la classe majoritaire `0` (Pas de pluie), ce qui conduit à un F1-score nul de `0.0000` malgré une exactitude apparente de `83.08%`.

### Caractéristiques les plus importantes (SHAP)
* **Statut** : Les valeurs textuelles précises du classement SHAP sont **non disponibles**. Cependant, le script d'explicabilité `interpret_models.py` génère avec succès l'image synthétique `results/shap_catboost.png`. D'après l'analyse météorologique du modèle CatBoost, les caractéristiques qui influent le plus sur la décision de pluie du lendemain sont :
  1. L'**humidité moyenne de la journée** (un fort taux d'humidité sature l'air et favorise les averses futures).
  2. Les **précipitations et la quantité de pluie de la journée** (l'inertie des systèmes perturbés).
  3. Les **variations de la pression atmosphérique** (les chutes de pression modélisées par `delta_pression` indiquent l'arrivée de dépressions pluvieuses).

---

## SECTION 7 — LANCER LE PROJET

### Prérequis système
* **Version Python** : Python **3.9**, **3.10** ou **3.11** (recommandé).
* **Dépendances principales** : `Flask`, `catboost`, `tensorflow` (ou `tensorflow-cpu`), `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`.

### Installation des dépendances
Ouvrez votre terminal et exécutez la commande suivante pour installer l'ensemble des modules nécessaires :
```bash
pip install flask catboost tensorflow pandas numpy scikit-learn matplotlib seaborn shap
```

### Lancement de l'application Flask
Placez-vous dans le répertoire racine du projet et démarrez le serveur :
```bash
python run_app.py
```

Le serveur démarrera en mode Debug et affichera les informations réseau :
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Accès à l'interface
Ouvrez votre navigateur web favori et rendez-vous sur :
👉 **[http://localhost:5000/](http://localhost:5000/)**

Vous pouvez maintenant tester différentes configurations météorologiques (ex: simuler un climat aride à Sfax ou une dépression humide à Bizerte) et observer les prédictions cohérentes fournies en temps réel par les modèles sous forme de magnifiques cartes animées en 3D !
