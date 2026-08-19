"""
Interface Streamlit — Prédiction de durée d'hospitalisation (DMS)

Prérequis : avoir exécuté train_model.py au préalable dans le même dossier
(génère model.pkl, columns.pkl, metadata.pkl).

Lancement : streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Chargement du modèle et des métadonnées (une seule fois, mis en cache)
# ---------------------------------------------------------------------------


@st.cache_resource
def charger_modele():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    metadata = joblib.load("metadata.pkl")
    return model, columns, metadata


model, columns, metadata = charger_modele()
MOTIFS = metadata["motifs"]
TRAITEMENTS = metadata["traitements"]

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Prédiction de la durée d'hospitalisation", page_icon="🏥")
st.image("https://img.magnific.com/vecteurs-libre/gens-qui-marchent-assis-au-batiment-hopital-exterieur-verre-clinique-ville-illustration-vectorielle-plane-pour-aide-medicale-urgence-architecture-concept-soins-sante_74855-10130.jpg?semt=ais_hybrid&w=740&q=80", use_container_width=True)
st.title("Prédiction de durée d'hospitalisation")
st.write(
    "Renseigner le profil du patient pour obtenir une estimation de la durée"
    "de séjour (en jours)"
)

st.header("Profil du patient")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        sexe = st.selectbox("Sexe", ["Féminin", "Masculin"])
with col2:
    with st.container(border=True):
        age = st.slider("Âge", min_value=18, max_value=98, value=50)

motif = st.selectbox("Motif d'hospitalisation", MOTIFS)

traitements_selectionnes = st.multiselect(
    "Traitements pré-hospitaliers en cours",
    TRAITEMENTS,
    help="Sélectionner tous les traitements que le patient prend déjà avant l'hospitalisation",
)

st.divider()

if st.button("Prédire la durée d'hospitalisation", type="primary"):
    # --- Construire le vecteur de features dans le même format que l'entraînement ---
    ligne = pd.DataFrame(0, index=[0], columns=columns)

    ligne["age"] = age
    ligne["sexe_encode"] = 1 if sexe == "Masculin" else 0
    ligne["nb_traitements"] = len(traitements_selectionnes)

    motif_col = f"motif_{motif}"
    if motif_col in ligne.columns:
        ligne[motif_col] = 1

    for t in traitements_selectionnes:
        traitement_col = f"traitement_{t}"
        if traitement_col in ligne.columns:
            ligne[traitement_col] = 1

    # --- Prédiction ---
    prediction = model.predict(ligne)[0]

    st.success(f"### Durée d'hospitalisation estimée : **{prediction:.1f} jours**")

    with st.expander("Détails du profil transmis au modèle"):
        st.write(f"**Sexe** : {sexe}")
        st.write(f"**Âge** : {age} ans")
        st.write(f"**Motif** : {motif}")
        st.write(
            f"**Traitements** : {', '.join(traitements_selectionnes) if traitements_selectionnes else 'Aucun'}"
        )

st.divider()
st.caption(
    "⚠️ Modèle entraîné sur des données synthétiques à des fins de prototype. "
    "Ne pas utiliser pour une décision clinique réelle."
)
