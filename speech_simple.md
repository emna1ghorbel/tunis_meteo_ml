# Discours Simple : Prédiction de Pluie en Tunisie

**SLIDE 1 — Titre**
Bonjour. Nous allons vous présenter notre projet pour prédire s'il va pleuvoir demain en Tunisie. Nous avons utilisé plusieurs méthodes de Machine Learning et créé une application web facile à utiliser.

**SLIDE 2 — Introduction & Problématique**
La Tunisie a des climats très différents, du nord au sud. Nous avons analysé 17 ans de données sur 4 villes. Le but est de deviner si demain, il y aura plus de 1 mm de pluie. Tout cela est accessible sur une interface web.

**SLIDE 3 — Le Dataset**
Nos données viennent de Bizerte, Tunis, Sousse et Sfax, de 2009 à 2026. Il y a 30 mesures par jour comme la température, le vent ou l'humidité. Le grand défi : il ne pleut que 17% du temps. Nous avons donc dû adapter nos modèles pour qu'ils n'ignorent pas la pluie.

**SLIDE 4 — Feature Engineering : Pourquoi ?**
Pourquoi faire du "Feature Engineering" ? Parce la météo d'aujourd'hui dépend de celle d'hier. Un jour de pluie n'arrive pas soudainement. Le modèle doit comprendre cette évolution temporelle. Nous avons donc créé de nouvelles variables pour lui donner une mémoire.

**SLIDE 5 — Feature Engineering : Lags & Rolling Stats**
Nous avons ajouté des variables de décalage : par exemple, s'il a plu hier ou il y a 7 jours. Nous avons aussi calculé des moyennes sur 7 et 30 jours pour voir si on est dans une période humide ou sèche. On fait cela ville par ville pour ne pas mélanger les données.

**SLIDE 6 — Feature Engineering : Features Dérivées & Encodage Cyclique**
Nous avons créé des indicateurs physiques : la variation de pression dans la journée ou la sécheresse du sol. Enfin, pour les mois, nous avons utilisé un encodage mathématique : cela aide le modèle à comprendre que décembre et janvier sont très proches en hiver.

**SLIDE 7 — Feature Learning : Vue d'ensemble**
Ensuite, le "Feature Learning". C'est laisser l'Intelligence Artificielle découvrir elle-même des relations cachées dans les données que nous n'avons pas vues. Nous avons utilisé trois méthodes : PCA, Autoencodeur et LSTM.

**SLIDE 8 — Feature Learning : PCA & Autoencodeur**
La PCA résume les variables de température en 3 composantes simples. L'Autoencodeur est un réseau qui compresse toutes les données d'une journée en seulement 16 valeurs clés. S'il peut résumer la journée avec 16 chiffres, c'est qu'ils sont très importants.

**SLIDE 9 — Feature Learning : LSTM sur 14 Jours**
Le LSTM analyse la météo sur les 14 derniers jours. Il lit cette période comme une histoire et en déduit une tendance avec 64 valeurs. Cela l'aide à comprendre si une dépression approche ou s'éloigne.

**SLIDE 10 — Vecteur Final : Pipeline Hybride**
À la fin, nous regroupons tout : les données de base, nos calculs manuels, et les résultats de nos réseaux de neurones. Chaque journée est donc décrite par un grand vecteur d'informations très complet. C'est ce qu'on donne au modèle final.

**SLIDE 11 — Les Modèles**
Notre modèle principal est CatBoost. Il gère très bien ce type de données et résiste bien au déséquilibre des classes, contrairement à un réseau de neurones classique qui avait tendance à prédire qu'il ne pleuvrait jamais.

**SLIDE 12 — Résultats**
Les résultats sont très bons. Sur de nouvelles données entre 2024 et 2026, CatBoost a obtenu un score AUC de 0.83. Il arrive bien à détecter les jours de pluie, prouvant que notre méthode fonctionne.

**SLIDE 13 — Interface Flask**
Pour finir, voici notre application web. À gauche, on entre la météo du jour. Quand on clique, la carte à droite tourne pour donner le résultat en 3D. S'il pleut, des gouttes tombent à l'écran. C'est simple, interactif et visuel.

**SLIDE 14 — Conclusion & Perspectives**
En conclusion, ce projet montre qu'en combinant notre compréhension de la météo et l'IA, on obtient d'excellentes prédictions. Plus tard, nous aimerions brancher l'application sur les données en temps réel de la météo tunisienne.

**SLIDE 15 — Questions / Réponses**
Merci de votre écoute ! Notre code est bien structuré et nous avons fait très attention à ne pas tricher avec les données. Avez-vous des questions sur notre démarche ou sur les modèles utilisés ?
