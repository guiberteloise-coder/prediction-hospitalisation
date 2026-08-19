# Nom du projet
<!-- ex : DMS-Predict / PredicHosp / etc. -->

## 📌 Description

Application permettant d'estimer la **durée d'hospitalisation (DMS)** d'un patient à partir de son profil (âge, sexe, motif d'hospitalisation, traitements en cours), à partir de données fictives générées synthétiquement mais avec des corrélations cliniquement plausibles.

**Objectif métier** : aider à une meilleure gestion des lits d'hôpitaux en anticipant la durée de séjour.

## 🎯 Fonctionnalités

- [✅] Génération / utilisation d'un dataset de patients fictifs avec corrélations réalistes
- [✅] Encodage des variables catégorielles
- [✅] Modèle de Machine Learning pour prédire la DMS
- [✅] Interface de saisie du profil patient
- [✅] Affichage de l'estimation
- [] Intervalle de confiance

## 🗂️ Structure du projet

```
.
├── generate_dataset.py            # Script de génération du dataset (Sprint 1 - Étape 1)
├── dataset_hospitalisation.csv    # Dataset généré (10 000 patients fictifs)
├── notebooks/                     # Notebooks d'exploration / entraînement (à venir)
├── src/                           # Code source (préprocessing, modèle, API...) (à venir)
├── app/                           # Interface (Streamlit)
├── requirements.txt
└── README.md
```

## ⚙️ Technologies utilisées

- Python
- Pandas / NumPy
- Scikit-learn / XGBoost
- Streamlit

## 🚀 Installation

```bash
git clone < https://github.com/guiberteloise-coder/prediction-hospitalisation>
cd <prediction_hospitalisation>
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
python generate_dataset.py

Lancement de l'application
streamlit run app/app.py
```

## 📊 Données

Le dataset dataset_hospitalisation.csv contient 10 000 patients fictifs générés de façon procédurale (seed fixée à 42 pour la reproductibilité), avec les colonnes suivantes :

Colonne	Type	Description
- id_patient : Identifiant unique du patient
- sexe	:	F / H
- age	:	Âge du patient (18 à 98 ans, moyenne ≈ 65 ans) — pyramide des âges pondérée vers les patients plus âgés, réaliste pour une population hospitalisée
- motif_hospitalisation	:	15 motifs possibles 
- traitements_pre_hospitaliers	:	Liste de traitements en cours avant l'hospitalisation (0 à 6 traitements), ou Aucun
- nb_traitements	:	Nombre de traitements pré-hospitaliers
- duree_hospitalisation_jours	:	Durée d'hospitalisation en jours (variable cible, DMS), de 1 à 29 jours dans le dataset actuel, moyenne ≈ 9,1 jours

Logique de génération de la DMS (voir generate_dataset.py) :

1. Durée de base tirée aléatoirement dans une fourchette propre à chaque motif
2. Modificateurs additifs selon les traitements en cours et l'âge du patient
3. Quelques interactions motif × traitement cliniquement notables (ex. anticoagulant + fracture, corticoïdes + sepsis)
4. Bruit gaussien final + clamp réaliste entre 1 et 45 jours

Les colonnes catégorielles (motif_hospitalisation, traitements_pre_hospitaliers) sont volontairement laissées non encodées dans ce CSV brut ; l'encodage est prévu à l'étape suivante du Sprint 1.

⚠️ Il s'agit de données entièrement **fictives**, générées synthétiquement — aucune donnée patient réelle n'est utilisée.

## 🧪 Méthodologie / Méthode Agile

Projet mené en sprints (méthode Agile) :
- ✅ Sprint 1 — Modèle de prédiction
- ✅ Sprint 2 — Interface utilisateur
- Sprint 3 — Rééquilibrer l'apprentissage sur les longs séjours
- Sprint 4 — Fiabilisation & amélioration

## 📈 Résultats

MAE : 1.78
RMSE : 2.26
R² : 0.71

## 🔮 Pistes d'amélioration

- Connexion à un vrai dossier patient informatisé (DPI)
- Enrichissement du dataset avec des corrélations plus réalistes (comorbidité multiples, complications,...)
- Ajout d'un intervalle de confiance sur la prédiction

## 👤 Auteur

Guibert Eloïse
