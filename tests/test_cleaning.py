"""Tests des filtres de nettoyage.

Trois tests sont des tests de non-régression : ils encodent des erreurs qui
étaient présentes dans la première version du travail et qui changeaient le
résultat du projet 1. Ils échouent si quelqu'un y revient.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gel.cleaning import (
    ajouter_horodatage,
    cumul_mesures_par_jour,
    masque_hors_saison_gel,
    masque_incoherence_abri,
    masque_journee_sous_echantillonnee,
    masque_site_usine,
    masque_valeurs_erreur,
    nettoyer,
)

PARAMETRES = {
    "valeurs_erreur_capteur": [60.0, -40.0],
    "mois_gel": [11, 12, 1, 2, 3, 4, 5],
    "ssi_max_nocturne": 50.0,
    "marge_humide_sur_seche": 0.5,
    "t2m_max": 10.0,
    "ecart_abri_attendu": 1.5,
    "ecart_abri_tolerance": 3.0,
    "min_mesures_par_jour": 50,
    "site_usine_lat": 47.266,
    "site_usine_lon": -1.561,
    "site_usine_rayon_deg": 0.05,
}


def jeu_minimal(**remplacements) -> pd.DataFrame:
    """Une ligne par défaut qui passe tous les filtres, modifiable champ par champ."""
    base = {
        "date": pd.Timestamp("2022-01-15"),
        "hour": 3,
        "station_id": 1000,
        "nbre_records": 5,
        "T2M": 2.0,
        "DEWT2M": 1.0,
        "RH2M": 93.0,
        "WS2M": 1.0,
        "WS10M": 1.4,
        "WD": 180.0,
        "VPD": 0.05,
        "CLDC": 40.0,
        "SSI": 0.0,
        "PRECIP": 0.0,
        "lat": 44.5,
        "lon": -0.6,
        "T2M_D_med": 0.6,
        "T2M_U_med": 0.2,
    }
    base.update(remplacements)
    return pd.DataFrame([base])


def test_valeurs_sentinelles_detectees_sur_les_deux_temperatures():
    df = pd.concat(
        [
            jeu_minimal(T2M_D_med=60.0),
            jeu_minimal(T2M_U_med=-40.0),
            jeu_minimal(T2M_D_med=59.9),
        ],
        ignore_index=True,
    )
    assert masque_valeurs_erreur(df, (60.0, -40.0)).tolist() == [True, True, False]


def test_saison_de_gel_couvre_novembre_a_mai():
    dates = pd.to_datetime(
        ["2021-11-05", "2021-12-20", "2022-01-10", "2022-05-31", "2022-06-01", "2022-10-31"]
    )
    df = pd.DataFrame({"date": dates})
    assert masque_hors_saison_gel(df, (11, 12, 1, 2, 3, 4, 5)).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_incoherence_abri_symetrique_autour_de_l_ecart_attendu():
    # L'ecart attendu est de 1,5 degre avec 3 degres de tolerance : on accepte
    # donc T2M - T2M_D_med entre -1,5 et 4,5.
    df = pd.concat(
        [
            jeu_minimal(T2M=2.0, T2M_D_med=0.6),  # ecart 1,4 : conserve
            jeu_minimal(T2M=2.0, T2M_D_med=-3.0),  # ecart 5,0 : rejete
            jeu_minimal(T2M=2.0, T2M_D_med=4.0),  # ecart -2,0 : rejete
        ],
        ignore_index=True,
    )
    assert masque_incoherence_abri(df, 1.5, 3.0).tolist() == [False, True, True]


def test_filtre_journalier_porte_sur_le_cumul_de_la_journee():
    """Non-regression : le seuil de 50 est un cumul journalier, pas un seuil horaire.

    La station 1 emet 10 fois par heure pendant 6 heures, soit 60 mesures : la
    journee est saine et doit etre conservee, bien qu'aucune heure n'atteigne 50.
    La station 2 emet 4 fois par heure pendant 5 heures, soit 20 mesures : la
    journee entiere doit disparaitre. Un filtre applique ligne a ligne
    (nbre_records < 4) donnerait exactement le resultat inverse.
    """
    lignes = [jeu_minimal(station_id=1, hour=h, nbre_records=10) for h in range(1, 7)]
    lignes += [jeu_minimal(station_id=2, hour=h, nbre_records=4) for h in range(1, 6)]
    df = pd.concat(lignes, ignore_index=True)

    cumuls = cumul_mesures_par_jour(df)
    assert set(cumuls["mesures_jour"]) == {60, 20}

    masque = masque_journee_sous_echantillonnee(df, cumuls, 50)
    assert masque[df["station_id"] == 1].sum() == 0
    assert masque[df["station_id"] == 2].all()


def test_site_usine_est_un_disque_et_non_une_egalite():
    """Non-regression : le filtre est une distance, pas une egalite de coordonnee.

    La troisieme station a exactement la latitude du site de Nantes mais se
    trouve a plus de 7 degres en longitude : une condition `lat == 47.266`
    l'aurait supprimee a tort.
    """
    df = pd.concat(
        [
            jeu_minimal(lat=47.2700, lon=-1.5600),  # dans le disque
            jeu_minimal(lat=47.3000, lon=-1.5610),  # a 0,034 degre : dans le disque
            jeu_minimal(lat=47.2660, lon=5.0000),  # meme latitude, tres loin
            jeu_minimal(lat=44.7000, lon=-0.6000),  # Bordeaux
        ],
        ignore_index=True,
    )
    assert masque_site_usine(df, 47.266, -1.561, 0.05).tolist() == [True, True, False, False]


def test_nettoyage_complet_journal_et_coherence():
    lignes = [jeu_minimal(station_id=1, hour=h, nbre_records=10) for h in range(1, 7)]
    lignes += [
        jeu_minimal(station_id=2, hour=2, nbre_records=10, date=pd.Timestamp("2022-07-15")),
        jeu_minimal(station_id=3, hour=2, nbre_records=10, SSI=120.0),
        jeu_minimal(station_id=4, hour=2, nbre_records=10, T2M=14.0, T2M_D_med=12.5),
        jeu_minimal(station_id=5, hour=2, nbre_records=10, T2M_U_med=float("nan")),
        jeu_minimal(station_id=6, hour=2, nbre_records=10, lat=47.27, lon=-1.56),
    ]
    df = pd.concat(lignes, ignore_index=True)

    propre, journal = nettoyer(df, PARAMETRES)

    assert len(journal) == 9, "les neuf filtres de l'enonce doivent etre traces"
    assert journal.attrs["lignes_initiales"] == len(df)
    assert journal["lignes_restantes"].iloc[-1] == len(propre)
    # Seules les six lignes saines de la station 1 survivent.
    assert set(propre["station_id"]) == {1}
    assert len(propre) == 6
    assert journal["lignes_supprimees"].sum() == len(df) - len(propre)


def test_horodatage_heure_24_tombe_a_minuit_du_jour_suivant():
    df = pd.concat(
        [jeu_minimal(hour=24, date=pd.Timestamp("2022-06-02")), jeu_minimal(hour=1)],
        ignore_index=True,
    )
    resultat = ajouter_horodatage(df)
    assert resultat["fin_fenetre"].iloc[0] == pd.Timestamp("2022-06-03 00:00")
    assert resultat["fin_fenetre"].iloc[1] == pd.Timestamp("2022-01-15 01:00")


@pytest.mark.parametrize("colonne", ["T2M_U_med", "T2M_D_med", "T2M", "SSI"])
def test_les_filtres_ne_modifient_aucune_valeur(colonne):
    """Le nettoyage supprime des lignes, il n'en corrige jamais les valeurs."""
    lignes = [jeu_minimal(station_id=1, hour=h, nbre_records=10) for h in range(1, 7)]
    df = pd.concat(lignes, ignore_index=True)
    propre, _ = nettoyer(df, PARAMETRES)
    pd.testing.assert_series_equal(
        propre[colonne], df.loc[df.index[: len(propre)], colonne], check_names=False
    )
