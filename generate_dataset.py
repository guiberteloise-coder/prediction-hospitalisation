"""
Sprint 1 — Étape 1 : génération du dataset hospitalier avec corrélations réalistes.

Objectif : produire un CSV "brut" (colonnes catégorielles en clair, PAS encodées)
dont la durée d'hospitalisation (DMS) est corrélée de façon plausible avec :
  - le motif d'hospitalisation
  - les traitements pré-hospitaliers en cours
  - l'âge du patient

Logique de calcul de la DMS :
  1. Durée de base tirée aléatoirement dans une fourchette propre à chaque motif
  2. Modificateurs additifs appliqués selon le profil (traitements + âge)
  3. Bruit gaussien final pour éviter un signal trop "propre"
  4. Clamp dans une fourchette réaliste [1, 45] jours

Rien n'est encodé ici : le motif et les traitements restent des chaînes de
caractères lisibles. L'encodage (one-hot, multi-label binarization...) est
volontairement laissé à l'étape suivante du Sprint 1.
"""

import random
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_PATIENTS = 10000

# ---------------------------------------------------------------------------
# Motifs d'hospitalisation : durée de base (min, max) en jours
# Fourchettes indicatives, cohérentes avec des ordres de grandeur cliniques
# ---------------------------------------------------------------------------

MOTIFS_DUREE_BASE = {
    "Fracture (membre inférieur)": (3, 8),
    "Fracture (membre supérieur)": (2, 5),
    "Pneumonie": (4, 10),
    "AVC": (7, 18),
    "Sepsis": (6, 15),
    "Décompensation cardiaque": (5, 12),
    "Appendicite": (2, 5),
    "Chirurgie programmée (prothèse hanche/genou)": (4, 9),
    "Douleurs abdominales à explorer": (2, 6),
    "Exacerbation BPCO": (4, 9),
    "Chute avec traumatisme": (2, 6),
    "Occlusion intestinale": (5, 12),
    "Infection urinaire compliquée": (3, 7),
    "Crise convulsive / épilepsie": (2, 5),
    "Embolie pulmonaire": (5, 11),
}

MOTIFS = list(MOTIFS_DUREE_BASE.keys())

# ---------------------------------------------------------------------------
# Traitements pré-hospitaliers (molécules de ville, non exhaustif mais réaliste)
# Chaque traitement a un "poids" de modification de la DMS (jours, avant bruit)
# ---------------------------------------------------------------------------

TRAITEMENTS_MODIFICATEURS = {
    # Anticoagulants — risque hémorragique, surveillance biologique
    "Anticoagulant (AVK)": (1.5, 3.0),
    "Anticoagulant (AOD)": (1.0, 2.5),
    # Antiagrégants — surveillance hémorragique moindre que les anticoagulants
    "Antiagrégant plaquettaire (aspirine)": (0.5, 1.0),
    "Antiagrégant plaquettaire (clopidogrel)": (0.5, 1.2),
    # Insuline — équilibration glycémique en hospitalisation
    "Insuline": (1.0, 2.0),
    # Antidiabétiques oraux — impact plus faible que l'insuline
    "Metformine": (0.2, 0.6),
    # Corticoïdes — immunosuppression, risque infectieux/décompensation
    "Corticoïdes (au long cours)": (1.0, 2.0),
    # Traitements cardio courants — impact faible, reflètent une comorbidité de fond
    "Bêta-bloquant": (0.2, 0.6),
    "IEC / ARA2": (0.2, 0.6),
    "Statine": (0.1, 0.4),
    # Respiratoire — pertinent surtout pour motifs pneumo, effet modéré
    "Bronchodilatateur inhalé": (0.3, 0.8),
    # Psychotropes — peuvent compliquer la prise en charge et la sortie
    "Antidépresseur": (0.3, 0.8),
    "Benzodiazépine": (0.3, 1.0),
    # Douleur chronique — pas d'effet direct marqué
    "Antalgique palier 1 (paracétamol)": (0.0, 0.2),
    # Thyroïde — effet négligeable
    "Lévothyroxine": (0.0, 0.2),
}

TRAITEMENTS = list(TRAITEMENTS_MODIFICATEURS.keys())

# Nombre de traitements pré-hospitaliers par patient (0 possible)
# Pondération réaliste : beaucoup de patients avec 0-2 traitements, peu avec 5+
NB_TRAITEMENTS_POSSIBLES = [0, 1, 2, 3, 4, 5, 6]
NB_TRAITEMENTS_POIDS = [0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03]


def tirer_traitements() -> list:
    """Tire un sous-ensemble aléatoire de traitements pré-hospitaliers."""
    nb = random.choices(NB_TRAITEMENTS_POSSIBLES, weights=NB_TRAITEMENTS_POIDS, k=1)[0]
    if nb == 0:
        return []
    return random.sample(TRAITEMENTS, k=nb)


def tirer_age() -> int:
    """Pyramide des âges réaliste pour une population hospitalisée
    (sur-représentation des patients âgés, mais toutes tranches présentes)."""
    tranche = random.choices(
        population=["18-40", "41-64", "65-79", "80+"],
        weights=[0.15, 0.25, 0.35, 0.25],
        k=1,
    )[0]
    bornes = {"18-40": (18, 40), "41-64": (41, 64), "65-79": (65, 79), "80+": (80, 98)}
    lo, hi = bornes[tranche]
    return random.randint(lo, hi)


def calculer_dms(motif: str, traitements: list, age: int) -> int:
    """Calcule la durée d'hospitalisation en appliquant la logique de corrélations.

    1) Base tirée dans la fourchette du motif
    2) + somme des modificateurs des traitements présents
    3) + modificateur lié à l'âge
    4) + bruit gaussien
    5) clamp [1, 45]
    """
    lo, hi = MOTIFS_DUREE_BASE[motif]
    duree = random.uniform(lo, hi)

    # --- Modificateurs traitements (cumulatifs) ---
    for t in traitements:
        mod_lo, mod_hi = TRAITEMENTS_MODIFICATEURS[t]
        duree += random.uniform(mod_lo, mod_hi)

    # Bonus polymédication : au-delà de 4 traitements, la fragilité globale
    # du patient pèse plus que la simple somme des effets individuels
    if len(traitements) >= 4:
        duree += random.uniform(0.5, 1.5)

    # Patient sans aucun traitement : a priori moins de comorbidités de fond
    if len(traitements) == 0:
        duree -= random.uniform(0.5, 1.5)

    # --- Modificateur âge ---
    if age >= 80:
        duree += random.uniform(1.5, 4.0)
    elif age >= 65:
        duree += random.uniform(0.5, 2.0)
    elif age < 40:
        duree -= random.uniform(0.5, 1.5)

    # --- Interactions motif × traitement (quelques cas cliniquement notables) ---
    if motif.startswith("Fracture") and any("Anticoagulant" in t for t in traitements):
        duree += random.uniform(1.0, 2.0)  # risque hémorragique péri-opératoire
    if motif == "AVC" and any(
        ("Anticoagulant" in t) or ("Antiagrégant" in t) for t in traitements
    ):
        duree += random.uniform(0.5, 1.5)
    if motif == "Sepsis" and any("Corticoïdes" in t for t in traitements):
        duree += random.uniform(1.0, 2.5)  # immunosuppression aggrave le pronostic

    # --- Bruit final ---
    duree += np.random.normal(loc=0, scale=1.2)

    # --- Clamp réaliste ---
    duree = max(1, min(45, duree))
    return round(duree)


def generer_patient(i: int) -> dict:
    sexe = random.choice(["F", "H"])
    age = tirer_age()
    motif = random.choice(MOTIFS)
    traitements = tirer_traitements()
    dms = calculer_dms(motif, traitements, age)

    return {
        "id_patient": f"{i+1:06d}",
        "sexe": sexe,
        "age": age,
        "motif_hospitalisation": motif,
        "traitements_pre_hospitaliers": ", ".join(traitements) if traitements else "Aucun",
        "nb_traitements": len(traitements),
        "duree_hospitalisation_jours": dms,
    }


def main():
    patients = [generer_patient(i) for i in range(N_PATIENTS)]
    df = pd.DataFrame(patients)

    output_path = "/content/dataset_hospitalisation.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Petit contrôle de cohérence des corrélations, affiché en console
    print("Aperçu :")
    print(df.head(10))
    print()
    print("DMS moyenne globale :", round(df["duree_hospitalisation_jours"].mean(), 2))
    print(
        "DMS moyenne SANS traitement :",
        round(df[df["nb_traitements"] == 0]["duree_hospitalisation_jours"].mean(), 2),
    )
    print(
        "DMS moyenne AVEC anticoagulant :",
        round(
            df[df["traitements_pre_hospitaliers"].str.contains("Anticoagulant")][
                "duree_hospitalisation_jours"
            ].mean(),
            2,
        ),
    )
    print(
        "DMS moyenne patients 80+ :",
        round(df[df["age"] >= 80]["duree_hospitalisation_jours"].mean(), 2),
    )
    print(
        "DMS moyenne patients <40 :",
        round(df[df["age"] < 40]["duree_hospitalisation_jours"].mean(), 2),
    )

    return output_path


if __name__ == "__main__":
    main()
