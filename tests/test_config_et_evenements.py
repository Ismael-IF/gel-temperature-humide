"""Tests de la configuration et de l'agrégation en évènements nocturnes."""

from __future__ import annotations

import pandas as pd
import pytest

from gel.config import charger_config
from gel.metrics import agreger_par_nuit, matrice_gel


def test_la_cible_est_bien_la_temperature_humide():
    """Non-regression : la variable a predire est T2M_U_med, pas T2M_D_med.

    La premiere version du travail avait entraine et evalue les modeles sur
    T2M_D_med, la temperature seche du capteur. Tout le projet portait donc sur
    la mauvaise variable. Ce test verrouille la cible dans la configuration.
    """
    cfg = charger_config()
    assert cfg.cible == "T2M_U_med"
    assert "T2M_D_med" not in cfg.features


def test_les_fenetres_temporelles_ne_se_chevauchent_pas():
    """L'apprentissage s'arrete avant le debut du test, sans recouvrement d'un jour."""
    cfg = charger_config()
    assert cfg.date("train_debut") < cfg.date("validation_debut") < cfg.date("train_fin")
    assert cfg.date("train_fin") < cfg.date("test_debut")


def test_les_dix_variables_de_l_enonce_sont_declarees():
    cfg = charger_config()
    attendues = {
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
    }
    assert set(cfg.features) == attendues


def test_agregation_par_nuit_regroupe_la_soiree_et_la_fin_de_nuit():
    """Une nuit va de midi a midi : 22 h et 5 h du matin suivant sont le meme evenement."""
    base = pd.DataFrame(
        {
            "station_id": [1, 1, 1, 1],
            "fin_fenetre": pd.to_datetime(
                [
                    "2023-04-10 22:00",
                    "2023-04-11 05:00",
                    "2023-04-11 23:00",
                    "2023-04-12 04:00",
                ]
            ),
            "T2M_U_med": [1.2, -0.8, 2.0, 0.5],
            "prediction": [1.0, -0.3, 2.2, 0.9],
        }
    )
    agrege = agreger_par_nuit(base)
    assert len(agrege) == 2
    premiere = agrege.iloc[0]
    assert premiere["heures"] == 2
    assert premiere["minimum_mesure"] == pytest.approx(-0.8)
    assert premiere["minimum_prediction"] == pytest.approx(-0.3)


def test_le_cout_horaire_surestime_le_nombre_d_evenements():
    """Non-regression : compter les erreurs a l'heure gonfle le cout economique.

    Une seule nuit de gel ratee, longue de six heures, doit compter pour un
    evenement manque et non pour six.
    """
    base = pd.DataFrame(
        {
            "station_id": [1] * 6,
            "fin_fenetre": pd.date_range("2023-04-10 22:00", periods=6, freq="h"),
            "T2M_U_med": [-0.5, -1.0, -1.5, -2.0, -1.2, -0.4],
            "prediction": [0.4, 0.3, 0.2, 0.5, 0.6, 0.8],
        }
    )
    horaire = matrice_gel(base["T2M_U_med"].to_numpy(), base["prediction"].to_numpy(), 0.0)
    agrege = agreger_par_nuit(base)
    evenement = matrice_gel(
        agrege["minimum_mesure"].to_numpy(), agrege["minimum_prediction"].to_numpy(), 0.0
    )
    assert horaire["faux_negatifs"] == 6
    assert evenement["faux_negatifs"] == 1
