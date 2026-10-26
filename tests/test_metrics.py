"""Tests des métriques d'erreur, de détection et de cout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gel.metrics import (
    balayage_seuil_alerte,
    cout_decision,
    matrice_gel,
    metriques_regression,
    taux_depassement,
)


def test_metriques_regression_sur_valeurs_connues():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, -1.0, 2.0, -2.0])
    resultat = metriques_regression(y, pred)
    assert resultat["rmse"] == pytest.approx(np.sqrt(2.5))
    assert resultat["mae"] == pytest.approx(1.5)
    assert resultat["biais"] == pytest.approx(0.0)


def test_rmse_et_mse_ne_sont_pas_confondues():
    """Non-regression : une MSE de 1,3 n'est pas une erreur moyenne de 1,3 degre.

    La premiere version du travail annoncait une MSE puis la commentait comme une
    erreur en degres. L'ecart est systematique : la racine d'une MSE superieure a
    1 est plus petite qu'elle.
    """
    y = np.zeros(100)
    pred = np.full(100, 1.5)
    resultat = metriques_regression(y, pred)
    mse = float(np.mean((pred - y) ** 2))
    assert mse == pytest.approx(2.25)
    assert resultat["rmse"] == pytest.approx(1.5)
    assert resultat["rmse"] != pytest.approx(mse)


def test_biais_est_signe_dans_le_sens_prediction_moins_mesure():
    """Un modele qui surestime la temperature humide sous-estime le risque de gel."""
    y = np.zeros(10)
    assert metriques_regression(y, y + 0.8)["biais"] == pytest.approx(0.8)
    assert metriques_regression(y, y - 0.8)["biais"] == pytest.approx(-0.8)


def test_taux_de_depassement_des_seuils():
    y = np.zeros(10)
    pred = np.array([0.0, 0.5, 0.9, 1.1, 1.5, 1.9, 2.1, 2.5, 3.0, 5.0])
    resultat = taux_depassement(y, pred, (1.0, 2.0))
    assert resultat["depassement_1C_pct"] == pytest.approx(70.0)
    assert resultat["depassement_2C_pct"] == pytest.approx(40.0)


def test_matrice_gel_cas_construit_a_la_main():
    # Observe : 4 heures de gel (valeurs negatives), 6 heures sans gel.
    y = np.array([-2.0, -1.0, -0.5, -0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Le modele n'annonce que 3 des 4 gelees, et declenche une fausse alerte.
    pred = np.array([-1.5, -0.8, -0.2, 0.3, -0.4, 1.2, 2.2, 3.1, 4.4, 5.5])
    resultat = matrice_gel(y, pred, 0.0)
    assert (resultat["vrais_positifs"], resultat["faux_negatifs"]) == (3, 1)
    assert (resultat["faux_positifs"], resultat["vrais_negatifs"]) == (1, 5)
    assert resultat["pod_rappel"] == pytest.approx(0.75)
    assert resultat["far_fausses_alertes"] == pytest.approx(0.25)
    assert resultat["csi"] == pytest.approx(3 / 5)


def test_relever_le_seuil_d_alerte_augmente_la_detection():
    """Alerter plus tot ne peut pas faire baisser le taux de detection."""
    rng = np.random.default_rng(1)
    y = rng.normal(1.0, 2.0, 5000)
    pred = y + rng.normal(0.0, 0.8, 5000)
    table = balayage_seuil_alerte(y, pred, np.arange(-1.0, 2.01, 0.25), 4000.0, 120.0)
    assert table["pod_rappel"].is_monotonic_increasing
    assert table["faux_positifs"].is_monotonic_increasing


def test_comparaison_par_nuit_compte_les_nuits_gagnees():
    """Un modele qui gagne presque toutes les nuits doit ressortir comme tel.

    Et un modele dont l'ecart moyen est tres petit devant la dispersion de cet
    ecart d'une nuit a l'autre ne doit pas etre presente comme meilleur.
    """
    from gel.metrics import comparaison_par_nuit

    rng = np.random.default_rng(4)
    nuits = np.repeat(np.arange(40), 10)
    y = rng.normal(0, 3, nuits.size)
    predictions = {
        "precis": y + rng.normal(0, 0.5, y.size),
        "imprecis": y + rng.normal(0, 2.0, y.size),
    }
    table = comparaison_par_nuit(y, predictions, nuits, reference="precis").set_index("modele")

    assert table.loc["precis", "nuits_gagnees"] >= 38
    assert table.loc["precis", "nuits_evaluees"] == 40
    assert table.loc["imprecis", "ecart_moyen_vs_reference"] > 0
    assert table.loc["precis", "ecart_moyen_vs_reference"] == pytest.approx(0.0)


def test_les_groupes_categoriels_gardent_leur_ordre_logique():
    """Les regimes de temperature doivent se suivre du plus froid au plus doux.

    Trie comme des chaines de caracteres, « < -3 » arriverait apres « 0 a 2 » et
    la figure donnerait a lire une progression qui n'existe pas.
    """
    from gel.metrics import metriques_par_groupe

    rng = np.random.default_rng(5)
    taille = 600
    mesure = rng.uniform(-6, 8, taille)
    etiquettes = ["< -3", "-3 à -1", "-1 à 0", "0 à 2", "2 à 5", "> 5"]
    df = pd.DataFrame(
        {
            "cible": mesure,
            "prediction": mesure + rng.normal(0, 1, taille),
            "regime": pd.cut(mesure, [-np.inf, -3, -1, 0, 2, 5, np.inf], labels=etiquettes),
        }
    )
    table = metriques_par_groupe(df, "regime", "cible", "prediction", minimum_observations=10)
    assert table["regime"].astype(str).tolist() == etiquettes


def test_cout_de_decision_est_la_somme_ponderee_des_erreurs():
    contingence = {"faux_negatifs": 3, "faux_positifs": 10}
    resultat = cout_decision(contingence, 4000.0, 120.0, surface_ha=2.0)
    assert resultat["cout_total_eur"] == pytest.approx(2 * (3 * 4000 + 10 * 120))
    assert resultat["cout_gelees_ratees_eur"] == pytest.approx(24000.0)


def test_asymetrie_des_couts_deplace_le_seuil_optimal():
    """Plus une gelee ratee coute cher, plus il faut alerter tot.

    C'est le resultat qui justifie de calibrer un seuil d'alerte plutot que de
    prendre 0 degre par defaut.
    """
    rng = np.random.default_rng(2)
    y = rng.normal(1.0, 2.0, 20000)
    pred = y + rng.normal(0.0, 0.9, 20000)
    seuils = np.arange(-1.0, 3.01, 0.1)

    prudent = balayage_seuil_alerte(y, pred, seuils, 10000.0, 100.0)
    econome = balayage_seuil_alerte(y, pred, seuils, 500.0, 100.0)
    seuil_prudent = prudent.loc[prudent["cout_total_eur"].idxmin(), "seuil_alerte"]
    seuil_econome = econome.loc[econome["cout_total_eur"].idxmin(), "seuil_alerte"]
    assert seuil_prudent > seuil_econome
