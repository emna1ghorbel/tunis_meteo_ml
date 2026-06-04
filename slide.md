# 🌧️ Présentation : Prédiction de Pluie en Tunisie
### Pipeline ML Hybride : Feature Engineering + Feature Learning + CatBoost + Deep Learning

Ce document contient pour chaque slide :
1. **Ce qui est affiché à l'écran** (les bullet points visuels de votre présentation)
2. **Le discours oral** (les phrases complètes à prononcer)

---

## SLIDE 1 — TITRE

**Contenu de la slide (Affiché à l'écran) :**
* **Titre :** Prédiction de Pluie en Tunisie
* **Sous-titre :** Pipeline ML Hybride : Feature Engineering + Feature Learning + CatBoost + Deep Learning
* **Points clés :** 17 ans de données réelles · 4 villes · Interface Web Flask

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 30 secondes]*
Bonjour à tous. Le projet que nous allons vous présenter aujourd'hui répond à une question concrète et importante : est-ce qu'il va pleuvoir demain en Tunisie ? Pour y répondre, nous avons construit un pipeline hybride complet qui combine l'expertise métier du feature engineering avec la puissance de l'apprentissage profond (feature learning) et de CatBoost. L'ensemble s'appuie sur 17 ans de données réelles et aboutit à une interface web interactive.

---

## SLIDE 2 — INTRODUCTION & PROBLÉMATIQUE

**Contenu de la slide (Affiché à l'écran) :**
* **Problématique :** Est-ce qu'il va pleuvoir demain en Tunisie ?
* **Contexte :** Diversité climatique remarquable : de Bizerte (méditerranéen humide) à Sfax (semi-aride)
* **Dataset :** 23 978 observations — 2009 à 2026, 4 villes
* **Variable cible :** `pluie_demain_bin` = 1 si précipitations > 1,0 mm le lendemain
* **Livrable :** Interface web Flask interactive avec prédiction animée en temps réel

**Discours (Ce que vous dites à l'oral) :**
Nous avons choisi la Tunisie parce qu'elle présente une diversité climatique remarquable sur un territoire relativement compact — du climat méditerranéen humide de Bizerte au nord jusqu'aux influences quasi-désertiques de Sfax au sud. Notre jeu de données couvre 17 ans de relevés météorologiques réels, de 2009 à 2026, soit près de 24 000 observations. La cible que nous cherchons à prédire est `pluie_demain_bin`, qui vaut 1 uniquement si les précipitations du lendemain dépassent 1,0 mm, c'est-à-dire une vraie pluie. Et pour rendre ce travail accessible, nous avons développé une interface web Flask interactive qui permet de saisir les conditions du jour et d'obtenir instantanément une prédiction animée.

---

## SLIDE 3 — LE DATASET

**Contenu de la slide (Affiché à l'écran) :**
* **Fichier :** `tunisie_meteo_reelle_2009_2026.csv`
* **Villes & Profils climatiques :**
  * *Bizerte :* Méditerranéen classique, hivers pluvieux
  * *Tunis :* Méditerranéen, températures plus chaudes
  * *Sousse :* Méditerranéen côtier, précipitations modérées
  * *Sfax :* Semi-aride, influence saharienne
* **30 variables brutes en 6 familles :** Température · Précipitations/Humidité · Vent · Pression · Ensoleillement · Bilan hydrique
* **Déséquilibre de classes :**
  * Classe 0 (pas de pluie) : 83,1%
  * Classe 1 (pluie) : 16,9%
* **Traitement :** Poids de classes inverses (LSTM) + seuil strict à 1,0 mm

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
Les données proviennent d'un dataset de 17 ans regroupant quatre grandes villes aux régimes climatiques très différents : Bizerte et Tunis au nord avec leurs hivers pluvieux, Sousse au centre, et Sfax au sud, plus aride. Nous disposons de 30 variables météorologiques brutes que l'on peut regrouper en familles : température, humidité, vent, pression atmosphérique, ensoleillement et bilan hydrique du sol. Le défi majeur de ce dataset est son fort déséquilibre de classes : il ne pleut que dans 16,9% des cas. Nous avons géré ce problème en appliquant des poids de classes inverses lors de l'entraînement de nos réseaux de neurones, pour forcer le modèle à ne pas ignorer ces événements pluvieux rares, tout en gardant un seuil de pluie strict à 1,0 mm pour éviter le bruit des simples traces d'humidité.

---

## SLIDE 4 — FEATURE ENGINEERING : POURQUOI ?

**Contenu de la slide (Affiché à l'écran) :**
* **L'idée centrale :** La météo a une mémoire — le modèle doit l'avoir aussi
* Un modèle ne voit qu'une ligne dans un tableau. Mais un jour de pluie n'arrive jamais seul :
  * La pression baisse progressivement
  * L'humidité monte sur plusieurs jours
  * Les systèmes frontaux se déplacent lentement
* **Solution :** Le feature engineering construit une mémoire artificielle pour le modèle (Implémenté dans `src/features.py`)

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 45 secondes]*
Avant d'expliquer nos transformations, il faut comprendre pourquoi elles sont nécessaires. Un modèle de machine learning standard ne voit qu'une chose : la ligne du jour actuel. Mais la météo a une mémoire. Un jour de pluie n'arrive jamais seul : il est précédé par une pression atmosphérique qui baisse sur plusieurs jours, une humidité qui monte, des fronts qui se déplacent. Si on ne donne au modèle que les mesures d'un seul jour, il est aveugle à cette dynamique. Le feature engineering, c'est l'art de donner au modèle cette mémoire artificielle, en lui construisant des variables qui résument ce qui s'est passé dans les jours précédents.

---

## SLIDE 5 — FEATURE ENGINEERING : LAGS & ROLLING STATS

**Contenu de la slide (Affiché à l'écran) :**
* **Lag features (décalages temporels) :**
  * `pluie_lag1`, `pluie_lag3`, `pluie_lag7` : Précipitations hier, il y a 3j, 7j
  * `temp_lag1`, `pression_lag1` : Température et pression d'hier
* **Rolling statistics (moyennes glissantes) :**
  * `rolling_pluie_7j` / `rolling_pluie_30j` → Régime perturbé ou anticyclone sec ?
  * `rolling_temp_7j` / `rolling_humidite_7j` → Contexte thermique et hydrique
* ⚠️ **Attention au leakage :** Tous les calculs sont groupés par ville (`groupby("ville")`) pour éviter tout mélange entre les séries temporelles de villes distinctes.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
Nous avons commencé par les lags, c'est-à-dire les valeurs passées. Nous avons créé des décalages sur 1, 3 et 7 jours pour la pluie, ainsi que pour la température et la pression d'hier. L'intuition est claire : s'il a plu hier, la pression est perturbée et la probabilité de pluie aujourd'hui augmente. Ensuite, nous avons calculé des moyennes glissantes sur 7 et 30 jours. Pourquoi ? Parce qu'un mois d'octobre avec une moyenne de 3 mm de pluie par jour n'est pas du tout le même contexte atmosphérique qu'un mois sec à 0 mm : la moyenne glissante capture la tendance de fond. Un point technique crucial : nous avons groupé tous ces calculs par ville avant de les décaler, pour éviter ce qu'on appelle le data leakage. Sans cela, le dernier jour de Bizerte deviendrait le "hier" du premier jour de Tunis, ce qui n'a aucun sens physique.

---

## SLIDE 6 — FEATURE ENGINEERING : FEATURES DÉRIVÉES & ENCODAGE CYCLIQUE

**Contenu de la slide (Affiché à l'écran) :**
* **Features dérivées à sens physique fort :**
  * `delta_pression` = `pression_max − pression_min` (Instabilité atmosphérique)
  * `stress_hydrique` = `evapotranspiration − precipitation` (Bilan du sol)
  * `ratio_ensoleillement` = `ensoleillement_h / duree_jour_h` (Couverture nuageuse)
* **Interactions saisonnières :** `saison_Hiver × pluie_lag1` ≠ `saison_Été × pluie_lag1`
* **Encodage cyclique du temps :**
  * Problème : Décembre (12) et Janvier (1) semblent éloignés de 11, alors qu'ils sont voisins.
  * Solution : `sin(2π × mois / 12)` et `cos(2π × mois / 12)`
  * Projection sur un cercle : Décembre et Janvier redeviennent proches.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute 15 secondes]*
Nous avons également créé des indicateurs physiques. Par exemple, l'amplitude de pression (`delta_pression`) : une chute de pression rapide dans la même journée annonce un front pluvieux. Le stress hydrique du sol : quand le sol perd plus d'eau qu'il n'en reçoit, l'air devient sec, ce qui modifie la convection. Enfin, nous avons un encodage cyclique du temps. Si on donne au modèle les mois sous forme numérique brute de 1 à 12, on lui ment, car il croira que décembre et janvier sont très différents (écart de 11) alors qu'ils sont en plein hiver tous les deux. En utilisant le sinus et le cosinus, nous projetons les mois sur un cercle unitaire : sur ce cercle, décembre et janvier sont à côté, tout comme dans la réalité climatique.

---

## SLIDE 7 — FEATURE LEARNING : VUE D'ENSEMBLE

**Contenu de la slide (Affiché à l'écran) :**
* **Pourquoi aller au-delà du feature engineering manuel ?**
  * L'ingénierie manuelle encode ce que l'on comprend de la physique.
  * Mais les données contiennent des structures complexes, non formulées explicitement.
* **Solution : Le Feature Learning**
  * Laisser des réseaux de neurones découvrir automatiquement des représentations utiles.
* **3 approches complémentaires (`src/feature_learning.py`) :**
  1. **PCA :** Débruitage des variables corrélées
  2. **Autoencodeur :** Représentation compressée de chaque journée
  3. **LSTM :** Dynamique temporelle sur fenêtre glissante de 14 jours

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 45 secondes]*
Malgré la richesse de notre feature engineering manuel, il y a une limite : nous, humains, ne pouvons pas anticiper toutes les interactions complexes cachées dans les données. Le feature learning, c'est laisser le modèle découvrir lui-même ces patterns. Nous avons combiné trois approches : la PCA pour réduire la redondance des corrélations, un autoencodeur pour trouver un résumé extrêmement compact d'une journée météo, et un réseau LSTM pour extraire la dynamique temporelle sur des fenêtres glissantes.

---

## SLIDE 8 — FEATURE LEARNING : PCA & AUTOENCODEUR

**Contenu de la slide (Affiché à l'écran) :**
* **PCA sur l'espace thermique :**
  * 6 variables de température corrélées (`temp_max`, `temp_min`, `temp_mean`, `rosee_max`...)
  * Compression en **3 composantes principales** indépendantes (`pca_1`, `pca_2`, `pca_3`)
  * Fit uniquement sur le train (≤ 2022) pour éviter le leakage.
* **Autoencodeur non supervisé :**
  * Architecture : `Input(30) → Dense(64) → Dense(32) → Dense(16, bottleneck) → Dense(32) → Dense(64) → Output(30)`
  * Objectif : Reconstruire les 30 variables après compression extrême.
  * Le *bottleneck* de **16 neurones** force un résumé essentiel → Features `ae_1` à `ae_16`.
  * Intuition : Si 16 chiffres suffisent à reconstruire une journée, ils en capturent l'essentiel.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
Premièrement, la PCA : nous avons 6 variables de température très corrélées. Elles disent presque la même chose. La PCA les compresse en 3 axes indépendants qui capturent l'essentiel de l'information sans redondance. Ensuite, l'autoencodeur : c'est un réseau de neurones dont le but est de reconstruire son entrée. L'astuce est de faire passer les 30 variables climatiques par un goulot d'étranglement de seulement 16 neurones. Ce qu'on garde, ce sont ces 16 activations. L'intuition est forte : si le réseau arrive à reconstruire une météo complexe à partir de 16 chiffres seulement, c'est que ce résumé est une représentation incroyablement dense et utile pour notre modèle final.

---

## SLIDE 9 — FEATURE LEARNING : LSTM SUR 14 JOURS

**Contenu de la slide (Affiché à l'écran) :**
* **Architecture LSTM supervisée :**
  * **Entrée :** Séquence de 14 jours × 30 variables standardisées
  * **Pourquoi 14 jours ?** Échelle typique d'un front méditerranéen (~2 semaines de l'Atlantique à la Tunisie)
* **Architecture :**
  * `LSTM(64, return_sequences=True) + Dropout(0.3)`
  * `LSTM(64) + Dropout(0.3)`
  * `Dense(1, sigmoid)`
* **Entraînement :**
  * Supervisé sur `pluie_demain_bin` avec poids de classes inversés
  * Extraction des 64 activations cachées (2ᵉ couche) → Features `lstm_1` à `lstm_64`
* ⚠️ **Attention :** Fenêtres pré-remplies à zéro aux frontières inter-villes pour ne jamais mélanger les historiques.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
L'autoencodeur ne voit qu'un instant, mais le LSTM, lui, lit l'histoire. Nous lui donnons des séquences de 14 jours. Pourquoi 14 jours ? Parce que c'est le temps typique qu'il faut à une dépression atlantique pour traverser la Méditerranée. Ce LSTM lit ces 14 jours, et produit un vecteur de 64 dimensions qui résume la dynamique passée : sort-on d'une période sèche ? La pression est-elle en chute sur la semaine ? Nous extrayons ces 64 neurones que nous ajoutons à notre jeu de données. Et là encore, nous sommes stricts sur le leakage : si une fenêtre de 14 jours déborde sur le début de l'historique d'une ville, on complète avec des zéros plutôt que d'emprunter les jours de la ville précédente.

---

## SLIDE 10 — VECTEUR FINAL : PIPELINE HYBRIDE

**Contenu de la slide (Affiché à l'écran) :**
* **Concaténation de toutes les sources** → `tunisie_meteo_final.csv`
* **Composition du vecteur par observation :**
  * Features brutes + engineerées (lags, rolling, dérivées, saisonnières) : ~45 variables
  * PCA thermique : 3 variables
  * Autoencodeur (bottleneck) : 16 variables
  * LSTM temporel : 64 variables
* **Résultat :** Un seul vecteur dense par jour = physiquement informé + algorithmiquement optimisé.
* C'est le principe du **pipeline hybride : Représentation + Prédiction**.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 45 secondes]*
Au final, comment assemble-t-on tout cela ? C'est simple, nous concaténons toutes ces sources ! Chaque journée météorologique est désormais décrite par les features brutes, les lags et indicateurs construits à la main, les 3 composantes PCA, les 16 embeddings de l'autoencodeur et les 64 embeddings du LSTM. C'est ce qu'on appelle un pipeline hybride représentation et prédiction. Le vecteur final donné au classifieur contient à la fois la physique métier et l'optimisation mathématique du deep learning.

---

## SLIDE 11 — LES MODÈLES

**Contenu de la slide (Affiché à l'écran) :**
* **CatBoost — Modèle principal**
  * Gestion native de la variable catégorielle `ville`
  * Robuste aux valeurs manquantes, pas de normalisation requise
  * Boosting ordonné → Réduit le surapprentissage temporel
  * *Hyperparamètres :* `iterations=500` · `learning_rate=0.05` · `depth=6` · `early_stopping=50`
* **Réseau Keras Feed-Forward — Comparaison**
  * Architecture : `Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → sigmoid`
  * ⚠️ Souffre d'un **collapse de classe** : prédit systématiquement "pas de pluie"
  * Conséquence : Accuracy apparente de 83%, mais F1-score = 0.00 sur la classe positive (Piège classique des données déséquilibrées).

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
Pour la prédiction finale, nous avons utilisé CatBoost. C'est un modèle parfait pour ce problème : il gère la variable ville nativement, il est robuste, et son boosting ordonné résiste bien au surapprentissage temporel. En parallèle, nous avons entraîné un réseau de neurones classique avec Keras pour comparer. Et c'est là qu'on observe un phénomène très intéressant : face aux données déséquilibrées (16% de pluie), le réseau Keras s'effondre. Il apprend qu'il a tout intérêt à prédire systématiquement "pas de pluie", ce qui lui donne une précision artificielle de 83%, mais il rate toutes les averses (F1-score de 0). CatBoost, lui, grâce à ses arbres, résiste très bien à ce déséquilibre.

---

## SLIDE 12 — RÉSULTATS

**Contenu de la slide (Affiché à l'écran) :**
* **Jeu de test :** Années 2024 à 2026 — 3 492 observations totalement indépendantes.
* **Performances sur le test :**
  * **CatBoost :** ROC AUC = **0.8335** | Accuracy = 84,82% | F1 classe positive = 0.41
  * **Keras FF :** ROC AUC = 0.7561 | Accuracy = ~83% | F1 classe positive = 0.00
* **Performance sur validation (2023) :**
  * CatBoost atteint un ROC AUC de **0.8727**.
* *Métriques disponibles dans :* `results/metrics_test.txt`, `results/metrics_validation.txt`, `results/model_metrics.json`

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 45 secondes]*
Les résultats le confirment. Sur notre jeu de test qui couvre les années 2024 à 2026, soit des données du futur que le modèle n'a jamais vues, CatBoost atteint un excellent ROC AUC de 0.8335 et un F1-score de 0.41 sur la classe positive, là où le réseau de neurones Keras s'effondre à 0. Sur les données de validation, CatBoost montait même à plus de 0.87 d'AUC. Ces métriques solides valident complètement notre approche hybride de features.

---

## SLIDE 13 — INTERFACE FLASK

**Contenu de la slide (Affiché à l'écran) :**
* **Ce que voit l'utilisateur (Deux colonnes) :**
* **Gauche — Formulaire de saisie :**
  * 6 paramètres : temp. max/min, précipitations, humidité, vent, pression.
  * Badge coloré dynamique par ville (Bizerte = turquoise, Tunis = rouge, Sousse = vert, Sfax = orange).
* **Droite — Carte de résultat animée :**
  * Rotation 3D à 360° pour révéler le verdict.
  * ☁️ **Pluie :** Carte bleu nuit, nuage SVG flottant, 45 gouttes animées en diagonale.
  * ☀️ **Soleil :** Carte orange, soleil SVG avec anneaux rotatifs.
  * Compteur luminescent s'incrémentant de 0% à la probabilité finale.
* **Flux technique :** JS calcule les dérivées → `fetch POST /predict` → Scoring thermodynamique (Flask) → Animation côté client.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 1 minute]*
Tout ce travail a été packagé dans une interface web très visuelle. L'utilisateur remplit à gauche six paramètres météo basiques. L'interface s'adapte même aux couleurs de la ville choisie. Quand on clique, le JavaScript calcule en temps réel les variables dérivées nécessaires et interroge notre serveur Flask. La carte de droite effectue alors une rotation 3D spectaculaire : si c'est la pluie, on a une ambiance bleu nuit avec des gouttes animées de façon fluide ; si c'est sec, une ambiance chaude et ensoleillée. Le pourcentage s'anime progressivement. Côté backend Flask, pour cette démo interactive, nous utilisons une formule de scoring thermodynamique physique, ultra-rapide, basée sur l'humidité, la pression et le vent.

---

## SLIDE 14 — CONCLUSION & PERSPECTIVES

**Contenu de la slide (Affiché à l'écran) :**
* **Ce que démontre ce projet :**
  * ✅ Feature engineering informé + feature learning profond → ROC AUC > 0.83.
  * ✅ Pipeline ML rigoureux sans aucun *data leakage*.
  * ✅ Interface utilisateur interactive de bout en bout.
* **Améliorations envisagées :**
  * Intégration des données en temps réel via l'API de l'INM (Institut National de Météorologie tunisien).
  * Déploiement en production : microservice API REST + ré-entraînement automatique mensuel.
  * Exploration d'architectures Transformer sur séquences météorologiques.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 45 secondes]*
En conclusion, nous avons prouvé qu'allier les connaissances physiques du feature engineering à la puissance du deep learning permet d'obtenir un AUC robuste de 0.83 sur des données réelles très déséquilibrées, et ce, en évitant le data leakage temporel à chaque étape. Pour la suite, les prochaines étapes logiques seraient de brancher cette application sur les API temps réel de l'INM tunisien, de conteneuriser le backend en microservices, et peut-être d'explorer les récents modèles de type Transformers temporels qui font des merveilles sur les séries météo de long terme.

---

## SLIDE 15 — QUESTIONS / RÉPONSES

**Contenu de la slide (Affiché à l'écran) :**
* **Principales questions anticipées :**
  * **Pourquoi CatBoost vs XGBoost ?** → Variables catégorielles natives, boosting ordonné, AUC 0.83 confirmé.
  * **Comment éviter le data leakage ?** → Fit uniquement sur ≤ 2022, `groupby("ville")` sur tous les lags.
  * **Pourquoi F1 = 0 pour Keras ?** → Collapse sur classe majoritaire, pas de `class_weight`, CatBoost y résiste mieux.
  * **Pourquoi 14 jours pour le LSTM ?** → Échelle des fronts méditerranéens (7j = trop court, 30j = bruit).
  * **Utilité de l'autoencodeur ?** → Capture les interactions non-linéaires implicites non formalisables manuellement.

**Discours (Ce que vous dites à l'oral) :**
*[DURÉE ESTIMÉE : 15 secondes]*
Merci à tous pour votre attention. Le code complet est structuré de manière professionnelle avec un pipeline scindé entre `src`, `models` et l'application `app`. Je suis maintenant à votre disposition si vous avez des questions sur l'implémentation de CatBoost, l'architecture de notre LSTM ou encore notre approche contre le data leakage !
