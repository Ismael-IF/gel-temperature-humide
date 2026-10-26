"""Variables explicatives, construites uniquement a partir des 10 variables autorisées.

L'énoncé impose une contrainte forte : la température humide doit être estimée
« uniquement en utilisant les variables T2M, DEWT2M, RH2M, WS2M, WS10M, WD,
VPD, CLDC, SSI, PRECIP ». Cette contrainte n'est pas administrative, elle
correspond au cas d'usage : si le modèle utilisait l'identifiant de station,
l'heure ou la position, il ne saurait plus servir une parcelle sans capteur,
qui est précisément la cible du produit.

Les transformations ci-dessous restent donc des fonctions de ces 10 variables.
Elles apportent deux choses :

- de la physique (pression de vapeur, écart au point de rosée, estimation de
  Stull), ce qui donne au modèle un point de départ correct au lieu de lui
  faire réapprendre la psychrométrie a partir de zéro ;
- un encodage correct de la direction du vent, qui est une variable circulaire :
  359 degrés et 1 degré sont voisins, alors qu'un arbre ou une régression les
  traitent comme deux extrêmes opposés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Les 10 variables atmosphériques autorisées par l'énoncé.
FEATURES_ENONCE: tuple[str, ...] = (
    "T2M",
    "DEWT2M",
    "RH2M",
    "WS2M",
    "WS10M",
    "WD",
    "VPD",
    "CLDC",
    "SSI",
    "PRECIP",
)

# Variables dérivées, toutes calculées a partir des précédentes.
FEATURES_DERIVEES: tuple[str, ...] = (
    "wd_sin",
    "wd_cos",
    "ecart_rosee",
    "pression_vapeur_sat",
    "pression_vapeur_reelle",
    "cisaillement_vent",
    "pluie",
    "tw_stull",
)

COEFFS_STULL_ORIGINE = (0.2831, 0.0003018, 0.01289, -4.0962)
EXPOSANT_STULL_ORIGINE = 0.2735


def pression_vapeur_saturante(temperature_c: np.ndarray | pd.Series) -> np.ndarray:
    """Pression de vapeur saturante en kPa (formule de Tetens).

    es(T) = 0,6108 * exp(17,27 T / (T + 237,3)). C'est la quantité physique qui
    relie température, humidité relative et déficit de saturation : la donner au
    modèle évite de lui faire approcher une exponentielle par morceaux.
    """
    t = np.asarray(temperature_c, dtype="float64")
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def temperature_humide_stull(
    t2m: np.ndarray | pd.Series,
    rh2m: np.ndarray | pd.Series,
    coeffs: tuple[float, float, float, float] = COEFFS_STULL_ORIGINE,
    exposant: float = EXPOSANT_STULL_ORIGINE,
) -> np.ndarray:
    """Formule de Stull simplifiée, telle que fournie dans l'énoncé.

    Tw = a * hr^p * Ts + b * hr^2 + c * hr + d

    avec Ts la température sèche sous abri (T2M), hr l'humidité relative en
    pourcentage, et les coefficients d'origine (a, b, c, d) et l'exposant p
    publiés par Stull. La formule est une corrélation empirique ajustée a
    pression standard : elle sert ici de référence sans paramètre, ce qui en
    fait la bonne base de comparaison pour les deux autres approches.
    """
    a, b, c, d = coeffs
    ts = np.asarray(t2m, dtype="float64")
    hr = np.clip(np.asarray(rh2m, dtype="float64"), 1.0, 100.0)
    return a * hr**exposant * ts + b * hr**2 + c * hr + d


def matrice_stull(
    t2m: np.ndarray | pd.Series,
    rh2m: np.ndarray | pd.Series,
    exposant: float = EXPOSANT_STULL_ORIGINE,
) -> np.ndarray:
    """Matrice de régression de la formule de Stull a exposant fixé.

    À exposant p fixé, la formule est **linéaire** en (a, b, c, d) : la
    recalibrer sur nos capteurs est donc un simple moindres carrés ordinaire,
    sans optimisation itérative ni risque de minimum local. Les colonnes
    renvoyées sont [hr^p * Ts, hr^2, hr, 1].
    """
    ts = np.asarray(t2m, dtype="float64")
    hr = np.clip(np.asarray(rh2m, dtype="float64"), 1.0, 100.0)
    return np.column_stack([hr**exposant * ts, hr**2, hr, np.ones_like(hr)])


def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées au DataFrame, sans toucher aux colonnes d'origine."""
    resultat = df.copy()

    # Direction du vent : variable circulaire, encodée en sinus / cosinus.
    radians = np.deg2rad(resultat["WD"].to_numpy(dtype="float64"))
    resultat["wd_sin"] = np.sin(radians).astype("float32")
    resultat["wd_cos"] = np.cos(radians).astype("float32")

    # Écart au point de rosée : premier déterminant du pouvoir évaporatif,
    # donc de la distance entre température sèche et température humide.
    resultat["ecart_rosee"] = (resultat["T2M"] - resultat["DEWT2M"]).astype("float32")

    es = pression_vapeur_saturante(resultat["T2M"])
    resultat["pression_vapeur_sat"] = es.astype("float32")
    resultat["pression_vapeur_reelle"] = (es * resultat["RH2M"].to_numpy() / 100.0).astype("float32")

    # Différence de vitesse entre 2 m et 10 m : indicateur de stratification de
    # l'air. Une nuit très stable (vent faible, fort gradient) est le contexte
    # typique d'une gelée radiative.
    resultat["cisaillement_vent"] = (resultat["WS10M"] - resultat["WS2M"]).astype("float32")

    # La pluie est très majoritairement nulle sur la saison de gel : la variable
    # binaire porte l'essentiel du signal et évite une queue de distribution
    # dominée par quelques épisodes.
    resultat["pluie"] = (resultat["PRECIP"] > 0).astype("int8")

    # Estimation physique de Stull, utilisée comme variable d'entrée : le modèle
    # n'a plus qu'a apprendre la correction entre l'abri a 2 m et le capteur
    # sans abri a 50 cm, ce qui est une tache bien plus simple.
    resultat["tw_stull"] = temperature_humide_stull(
        resultat["T2M"], resultat["RH2M"]
    ).astype("float32")

    return resultat


def jeux_de_variables() -> dict[str, list[str]]:
    """Jeux de variables mis en concurrence, du plus contraint au plus riche.

    `stull` reprend les seules entrées de la formule physique, ce qui permet de
    mesurer ce qu'apporte le fait d'ajouter des variables plutot que de changer
    de famille de modèle.
    """
    return {
        "stull": ["T2M", "RH2M"],
        "psychrometrique": ["T2M", "DEWT2M", "RH2M", "VPD", "ecart_rosee"],
        "psychrometrique_vent": [
            "T2M",
            "DEWT2M",
            "RH2M",
            "VPD",
            "ecart_rosee",
            "WS2M",
            "CLDC",
        ],
        "enonce": list(FEATURES_ENONCE),
        "complet": [c for c in FEATURES_ENONCE if c != "WD"] + list(FEATURES_DERIVEES),
    }
