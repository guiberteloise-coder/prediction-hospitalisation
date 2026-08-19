"""
Sprint 2 — Entraînement du modèle pour l'app Streamlit.

Charge dataset_hospitalisation.csv (généré au Sprint 1), encode les variables
catégorielles, entraîne un XGBoost (meilleur modèle retenu au Sprint 1) et
sauvegarde tout ce dont l'app Streamlit a besoin pour faire des prédictions :

  - model.pkl     : le modèle entraîné
  - columns.pkl   : la liste des colonnes attendues en entrée, dans le bon ordre
  - metadata.pkl  : la liste des motifs et des traitements possibles (pour les
                    menus déroulants de l'app)

A exécuter une seule fois (ou à chaque fois que le dataset / la logique de
corrélations change).

Utilisation : python train_model.py
"""

import joblib
import pandas as pd
from xgboost import XGBRegressor

SEED = 42
CSV_PATH = "dataset_hospitalisation.csv"

# ---------------------------------------------------------------------------
# 1. Chargement du dataset
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

MOTIFS = sorted(df["motif_hospitalisation"].unique().tolist())

# Liste de tous les traitements possibles (en dépliant la colonne, qui contient
# des listes séparées par ", ", "Aucun" signifiant aucun traitement)
traitements_set = set()
for val in df["traitements_pre_hospitaliers"]:
    if val != "Aucun":
        for t in val.split(", "):
            traitements_set.add(t)
TRAITEMENTS = sorted(traitements_set)

# ---------------------------------------------------------------------------
# 2. Encodage des variables catégorielles (PAS d'encodage dans le CSV brut)
# ---------------------------------------------------------------------------

df_encode = pd.DataFrame(index=df.index)

df_encode["age"] = df["age"]
df_encode["sexe_encode"] = (df["sexe"] == "H").astype(int)
df_encode["nb_traitements"] = df["nb_traitements"]

# One-hot du motif
for motif in MOTIFS:
    df_encode[f"motif_{motif}"] = (df["motif_hospitalisation"] == motif).astype(int)

# Multi-label binarisation des traitements
for traitement in TRAITEMENTS:
    df_encode[f"traitement_{traitement}"] = df["traitements_pre_hospitaliers"].apply(
        lambda val: int(traitement in val.split(", ")) if val != "Aucun" else 0
    )

df_encode["duree_hospitalisation_jours"] = df["duree_hospitalisation_jours"]

# ---------------------------------------------------------------------------
# 3. Entraînement XGBoost sur l'ensemble du dataset
# ---------------------------------------------------------------------------

X = df_encode.drop(columns=["duree_hospitalisation_jours"])
y = df_encode["duree_hospitalisation_jours"]

model = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED, n_jobs=-1
)
model.fit(X, y)

# ---------------------------------------------------------------------------
# 4. Sauvegarde du modèle et des métadonnées
# ---------------------------------------------------------------------------

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
joblib.dump({"motifs": MOTIFS, "traitements": TRAITEMENTS}, "metadata.pkl")

print("Modèle entraîné et sauvegardé : model.pkl, columns.pkl, metadata.pkl")
print(f"Nombre de colonnes en entrée : {len(X.columns)}")
