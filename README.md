# Nom du projet
<!-- ex : DMS-Predict / PredicHosp / etc. -->

## 📌 Description

Application permettant d'estimer la **durée d'hospitalisation (DMS)** d'un patient à partir de son profil (âge, sexe, motif d'hospitalisation, traitements en cours), à partir de données médicales anonymisées/fictives.

**Objectif métier** : aider à une meilleure gestion des lits d'hôpitaux en anticipant la durée de séjour.

## 🎯 Fonctionnalités

- [ ] Génération / utilisation d'un dataset de patients fictifs (CSV)
- [ ] Modèle de Machine Learning pour prédire la DMS
- [ ] Interface de saisie du profil patient
- [ ] Affichage de l'estimation + intervalle de confiance
- [ ] (Optionnel) Explicabilité des prédictions (facteurs les plus influents)
- [ ] (Optionnel) Tableau de bord de gestion des lits

## 🗂️ Structure du projet

```
.
├── data/               # Dataset(s) - CSV patients fictifs
├── notebooks/          # Notebooks d'exploration / entraînement
├── src/                # Code source (préprocessing, modèle, API...)
├── app/                # Interface (Streamlit / Flask...)
├── tests/               
├── requirements.txt
└── README.md
```
<!-- Adapte selon ton arborescence réelle -->

## ⚙️ Technologies utilisées

- Python
- Pandas / NumPy
- Scikit-learn / XGBoost (modèle ML)
- Streamlit ou Flask (interface)
- (Faker – utilisé initialement pour la locale FR, données générées via `random`)

## 🚀 Installation

```bash
git clone <url-du-repo>
cd <nom-du-dossier>
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
# Exemple de commande pour lancer l'app
python app/main.py
# ou
streamlit run app/app.py
```

## 📊 Données

Description du dataset : nombre de patients, colonnes (sexe, année de naissance, motif d'hospitalisation, traitements en cours, durée d'hospitalisation), et précision que les données sont **fictives/anonymisées**.

## 🧪 Méthodologie / Méthode Agile

Projet mené en sprints (méthode Agile) :
- Sprint 1 — Modèle de prédiction
- Sprint 2 — Interface utilisateur
- Sprint 3 — Fiabilisation & explicabilité

## 📈 Résultats

<!-- Métriques du modèle : MAE, RMSE, etc. -->

## 🔮 Pistes d'amélioration

- Connexion à un vrai dossier patient informatisé (DPI)
- Enrichissement du dataset avec des corrélations plus réalistes

## 👤 Auteur

<!-- Ton nom / contact -->

## 📄 Licence

<!-- MIT, etc. -->
