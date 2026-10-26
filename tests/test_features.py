"""Tests des variables explicatives et de la physique embarquée."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gel.features import (
    FEATURES_DERIVEES,
    FEATURES_ENONCE,
    construire_features,
    jeux_de_variables,
    matrice_stull,
    pression_vapeur_saturante,
    temperature_humide_stull,
)


def test_pression_vapeur_saturante_valeurs_de_reference():
    # Valeurs de table : 0,611 kPa a 0 degre, 2,34 kPa a 20 degres.
    assert pression_vapeur_saturante(np.array([0.0]))[0] == pytest.approx(0.6108, abs=1e-3)
    assert pression_vapeur_saturante(np.array([20.0]))[0] == pytest.approx(2.338, abs=0.01)


def test_stull_respecte_les_bornes_physiques():
    """La temperature humide est encadree par le point de rosee et la temperature seche."""
    rng = np.random.default_rng(0)
    t2m = rng.uniform(-10, 10, 2000)
    rh2m = rng.uniform(30, 100, 2000)
    tw = temperature_humide_stull(t2m, rh2m)
    # Tw ne peut pas depasser Ts (marge numerique pour la saturation a 100 %).
    assert np.all(tw <= t2m + 0.35)


def test_stull_est_croissante_avec_l_humidite():
    """À temperature fixee, plus l'air est humide, plus la temperature humide monte."""
    humidites = np.linspace(40, 100, 25)
    tw = temperature_humide_stull(np.full_like(humidites, 3.0), humidites)
    assert np.all(np.diff(tw) > 0)


def test_stull_saturation_rejoint_la_temperature_seche():
    """À 100 % d'humidite, temperature humide et temperature seche se confondent."""
    for ts in (-5.0, 0.0, 5.0, 10.0):
        tw = temperature_humide_stull(np.array([ts]), np.array([100.0]))[0]
        assert tw == pytest.approx(ts, abs=0.35)


def test_matrice_stull_reconstruit_la_formule():
    """La matrice de regression et la formule doivent donner le meme resultat.

    C'est ce qui autorise a recalibrer la formule par moindres carres ordinaires
    au lieu d'un ajustement non lineaire.
    """
    t2m = np.array([-3.0, 0.5, 7.2])
    rh2m = np.array([55.0, 88.0, 97.0])
    coeffs = np.array([0.2831, 0.0003018, 0.01289, -4.0962])
    attendu = temperature_humide_stull(t2m, rh2m)
    assert matrice_stull(t2m, rh2m) @ coeffs == pytest.approx(attendu, abs=1e-9)


def test_humidite_nulle_ne_fait_pas_exploser_la_formule():
    """RH2M vaut 1 % au minimum dans les donnees : la formule reste definie."""
    assert np.isfinite(temperature_humide_stull(np.array([5.0]), np.array([0.0]))).all()


def test_encodage_circulaire_du_vent():
    df = _jeu(WD=[0.0, 90.0, 180.0, 360.0])
    resultat = construire_features(df)
    assert resultat["wd_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert resultat["wd_sin"].iloc[1] == pytest.approx(1.0, abs=1e-6)
    # 0 et 360 degres designent la meme direction : leur encodage doit coincider.
    assert resultat["wd_sin"].iloc[0] == pytest.approx(resultat["wd_sin"].iloc[3], abs=1e-6)
    assert resultat["wd_cos"].iloc[0] == pytest.approx(resultat["wd_cos"].iloc[3], abs=1e-6)


def test_construire_features_ajoute_toutes_les_variables_derivees():
    resultat = construire_features(_jeu())
    assert set(FEATURES_DERIVEES).issubset(resultat.columns)
    assert set(FEATURES_ENONCE).issubset(resultat.columns)


def test_aucun_jeu_de_variables_ne_fuit_la_mesure_du_capteur():
    """Non-regression : la cible et la temperature seche a 50 cm sont interdites en entree.

    La premiere version du travail avait entraine le modele avec T2M_D_med, qui
    est une mesure du meme capteur sans abri que la cible. Un modele qui la
    recoit ne peut pas servir une parcelle non equipee, ce qui est justement le
    cas d'usage.
    """
    interdites = {"T2M_U_med", "T2M_D_med"}
    for nom, colonnes in jeux_de_variables().items():
        assert not interdites & set(colonnes), f"fuite de cible dans le jeu {nom}"


def _jeu(**remplacements) -> pd.DataFrame:
    base = {
        "T2M": [2.0],
        "DEWT2M": [1.0],
        "RH2M": [93.0],
        "WS2M": [1.0],
        "WS10M": [1.4],
        "WD": [180.0],
        "VPD": [0.05],
        "CLDC": [40.0],
        "SSI": [0.0],
        "PRECIP": [0.0],
    }
    base.update({cle: valeur for cle, valeur in remplacements.items()})
    longueur = max(len(valeur) for valeur in base.values())
    base = {cle: valeur * longueur if len(valeur) == 1 else valeur for cle, valeur in base.items()}
    return pd.DataFrame(base)
