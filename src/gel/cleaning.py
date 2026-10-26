"""Nettoyage du jeu de données horaires (projet 1).

L'énoncé liste neuf filtres. Ils sont implémentés ici comme autant de fonctions
qui renvoient le masque des lignes **à supprimer**, ce qui rend chaque règle
testable isolément et permet de produire un journal de nettoyage chiffré.

Deux points méritent une attention particulière, parce qu'ils sont faciles à
rater et qu'ils changent le résultat :

1. Le filtre de santé capteur porte sur le **cumul journalier de mesures par
   station**, pas sur le nombre de mesures d'une heure donnée. Il doit de plus
   être évalué sur les données brutes : si on le calcule après avoir retiré les
   heures diurnes, toutes les journées paraissent sous-échantillonnées et le
   filtre devient destructeur.
2. Le filtre du site usine est une **distance** au point de Nantes
   (47,266 ; -1,561), tolérance 0,05 degré, et non une égalité stricte sur la
   latitude ou la longitude : aucune station n'a de coordonnées exactement
   égales à celles du point de référence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Masque = pd.Series


@dataclass
class Filtre:
    """Une règle de nettoyage nommée, documentée et testable."""

    nom: str
    description: str
    fonction: Callable[[pd.DataFrame], Masque]


# --------------------------------------------------------------------------
# Règles élémentaires
# --------------------------------------------------------------------------
def masque_valeurs_erreur(df: pd.DataFrame, valeurs: tuple[float, ...] = (60.0, -40.0)) -> Masque:
    """+60 et -40 exactement : codes d'erreur du capteur, pas des températures."""
    masque = pd.Series(False, index=df.index)
    for colonne in ("T2M_D_med", "T2M_U_med"):
        for valeur in valeurs:
            masque |= df[colonne].eq(valeur)
    return masque


def masque_hors_saison_gel(
    df: pd.DataFrame, mois_gel: tuple[int, ...] = (11, 12, 1, 2, 3, 4, 5)
) -> Masque:
    """Hors novembre - mai : la question du gel ne se pose pas."""
    return ~df["date"].dt.month.isin(mois_gel)


def masque_diurne(df: pd.DataFrame, ssi_max: float = 50.0) -> Masque:
    """Rayonnement supérieur a 50 W/m2 : le capteur sans abri chauffe au soleil."""
    return df["SSI"] > ssi_max


def masque_humide_sur_seche(df: pd.DataFrame, marge: float = 0.5) -> Masque:
    """Tw > Ts + marge : physiquement impossible, donc mesure fausse."""
    return df["T2M_U_med"] > df["T2M_D_med"] + marge


def masque_trop_chaud(df: pd.DataFrame, t2m_max: float = 10.0) -> Masque:
    """Au-dela de 10 degrés sous abri, la situation est hors sujet gel."""
    return df["T2M"] > t2m_max


def masque_incoherence_abri(df: pd.DataFrame, ecart: float = 1.5, tolerance: float = 3.0) -> Masque:
    """Écart anormal entre la température sous abri (2 m) et le capteur (50 cm)."""
    return (df["T2M"] - df["T2M_D_med"] - ecart).abs() > tolerance


def masque_cible_manquante(df: pd.DataFrame, cible: str = "T2M_U_med") -> Masque:
    """Cible absente : la ligne n'est exploitable ni en apprentissage ni en test.

    L'énoncé mentionne aussi T2M_U_min et T2M_U_max. L'extrait fourni ne
    contient que la médiane horaire : le filtre porte donc sur les colonnes
    réellement présentes, ce qui est tracé dans le journal de nettoyage.
    """
    return df[cible].isna()


def cumul_mesures_par_jour(df: pd.DataFrame) -> pd.DataFrame:
    """Cumul de mesures par station et par jour, calculé sur les données brutes."""
    return (
        df.groupby(["station_id", "date"], observed=True)["nbre_records"]
        .sum()
        .rename("mesures_jour")
        .reset_index()
    )


def masque_journee_sous_echantillonnee(
    df: pd.DataFrame, cumuls: pd.DataFrame, minimum: int = 50
) -> Masque:
    """Journées station ou le capteur a émis strictement moins de `minimum` fois.

    Un capteur sain produit 96 a 240 mesures par jour. En dessous de 50, la
    médiane horaire n'est plus fiable : la journée entière est écartée.
    """
    suspects = cumuls.loc[cumuls["mesures_jour"] < minimum, ["station_id", "date"]]
    if suspects.empty:
        return pd.Series(False, index=df.index)
    cle = pd.MultiIndex.from_frame(df[["station_id", "date"]])
    cle_suspecte = pd.MultiIndex.from_frame(suspects)
    return pd.Series(cle.isin(cle_suspecte), index=df.index)


def masque_site_usine(
    df: pd.DataFrame, lat: float = 47.266, lon: float = -1.561, rayon: float = 0.05
) -> Masque:
    """Stations a moins de `rayon` degré du site de fabrication et de calibration.

    La distance est calculée en degrés, comme le formule l'énoncé. À cette
    latitude, 0,05 degré vaut environ 5,6 km en latitude et 3,8 km en longitude :
    le disque en degrés est donc un peu plus permissif en longitude qu'un vrai
    rayon métrique, ce qui va dans le sens de la prudence puisque l'objectif est
    d'exclure un site connu.
    """
    distance = np.hypot(df["lat"] - lat, df["lon"] - lon)
    return distance < rayon


# --------------------------------------------------------------------------
# Enchainement et journal
# --------------------------------------------------------------------------
def construire_filtres(parametres: dict) -> list[Filtre]:
    """Instancie les filtres de l'énoncé a partir de la configuration."""
    valeurs = tuple(parametres["valeurs_erreur_capteur"])
    mois = tuple(parametres["mois_gel"])
    ssi_max = float(parametres["ssi_max_nocturne"])
    marge = float(parametres["marge_humide_sur_seche"])
    t2m_max = float(parametres["t2m_max"])
    ecart = float(parametres["ecart_abri_attendu"])
    tolerance = float(parametres["ecart_abri_tolerance"])
    return [
        Filtre(
            "valeurs_erreur_capteur",
            "Températures sèche ou humide exactement égales aux codes d'erreur",
            lambda df: masque_valeurs_erreur(df, valeurs),
        ),
        Filtre(
            "hors_saison_gel",
            "Mois hors de la saison de gel (novembre a mai)",
            lambda df: masque_hors_saison_gel(df, mois),
        ),
        Filtre(
            "diurne",
            "Rayonnement solaire supérieur au seuil nocturne",
            lambda df: masque_diurne(df, ssi_max),
        ),
        Filtre(
            "humide_superieure_seche",
            "Température humide supérieure a la sèche, au-dela de la marge de mesure",
            lambda df: masque_humide_sur_seche(df, marge),
        ),
        Filtre(
            "trop_chaud",
            "Température sous abri trop élevée pour une situation de gel",
            lambda df: masque_trop_chaud(df, t2m_max),
        ),
        Filtre(
            "incoherence_abri",
            "Écart abri (2 m) / capteur (50 cm) hors tolérance",
            lambda df: masque_incoherence_abri(df, ecart, tolerance),
        ),
        Filtre(
            "cible_manquante",
            "Température humide manquante",
            masque_cible_manquante,
        ),
        Filtre(
            "site_usine",
            "Stations en fabrication, calibration ou test (site de Nantes)",
            lambda df: masque_site_usine(
                df,
                float(parametres["site_usine_lat"]),
                float(parametres["site_usine_lon"]),
                float(parametres["site_usine_rayon_deg"]),
            ),
        ),
    ]


def nettoyer(df: pd.DataFrame, parametres: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique les filtres de l'énoncé et renvoie (données nettoyées, journal).

    Le journal donne pour chaque règle deux chiffres volontairement distincts :

    - `lignes_supprimees` : effet marginal de la règle dans l'enchainement,
    - `lignes_concernees_seule` : nombre de lignes que la règle supprimerait si
      elle était appliquée seule au jeu brut.

    L'écart entre les deux mesure le recouvrement entre règles. C'est ce qui
    permet de dire quelle règle porte réellement la réduction de volume, et pas
    seulement laquelle a été appliquée en premier.
    """
    cumuls = cumul_mesures_par_jour(df)
    minimum = int(parametres["min_mesures_par_jour"])

    filtres = construire_filtres(parametres)
    filtres.insert(
        7,
        Filtre(
            "journee_sous_echantillonnee",
            "Journées station totalisant moins de mesures que le minimum attendu",
            lambda frame: masque_journee_sous_echantillonnee(frame, cumuls, minimum),
        ),
    )

    lignes_initiales = len(df)
    journal: list[dict] = []
    courant = df

    for filtre in filtres:
        seule = int(filtre.fonction(df).sum())
        masque = filtre.fonction(courant)
        supprimees = int(masque.sum())
        courant = courant.loc[~masque]
        journal.append(
            {
                "filtre": filtre.nom,
                "description": filtre.description,
                "lignes_supprimees": supprimees,
                "part_supprimee_pct": round(100 * supprimees / lignes_initiales, 3),
                "lignes_concernees_seule": seule,
                "lignes_restantes": len(courant),
            }
        )

    journal_df = pd.DataFrame(journal)
    journal_df.attrs["lignes_initiales"] = lignes_initiales
    return courant.reset_index(drop=True), journal_df


def ajouter_horodatage(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'instant de fin de la fenêtre horaire agrégée.

    Convention de l'énoncé : la mesure datée du 2 juin, heure 24, décrit ce qui
    s'est passé entre 23 h et minuit, soit la fin de la journée du 2 juin. On a
    donc `fin_fenetre = date + hour heures`, et l'heure 24 d'un jour tombe a
    minuit du jour suivant. Sans cette convention, les agrégations horaires sont
    décalées d'une heure et les comparaisons jour / nuit deviennent fausses.
    """
    resultat = df.copy()
    resultat["fin_fenetre"] = resultat["date"] + pd.to_timedelta(resultat["hour"], unit="h")
    return resultat
