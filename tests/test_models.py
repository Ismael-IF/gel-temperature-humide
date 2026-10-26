"""Tests des modèles : recalibrage de Stull, références, sélection polynomiale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gel.features import COEFFS_STULL_ORIGINE, temperature_humide_stull
from gel.models import (
    Persistance2m,
    StullBrut,
    StullRecalibre,
    construire_polynomiale,
    selectionner_polynomiale,
)


@pytest.fixture
def donnees_synthetiques() -> pd.DataFrame:
    """Jeu construit a partir de la formule de Stull, plus un bruit faible."""
    generateur = np.random.default_rng(7)
    t2m = generateur.uniform(-8, 10, 6000)
    rh2m = generateur.uniform(40, 100, 6000)
    cible = temperature_humide_stull(t2m, rh2m) + generateur.normal(0, 0.05, 6000)
    return pd.DataFrame({"T2M": t2m, "RH2M": rh2m, "T2M_U_med": cible})


def test_persistance_renvoie_la_temperature_sous_abri():
    df = pd.DataFrame({"T2M": [1.0, -2.0, 7.5]})
    assert Persistance2m().fit(df).predict(df).tolist() == [1.0, -2.0, 7.5]


def test_stull_brut_ne_depend_pas_des_donnees_d_apprentissage(donnees_synthetiques):
    """Le modele sans apprentissage doit predire la meme chose avant et apres fit."""
    modele = StullBrut()
    avant = modele.predict(donnees_synthetiques)
    modele.fit(donnees_synthetiques, donnees_synthetiques["T2M_U_med"])
    np.testing.assert_allclose(avant, modele.predict(donnees_synthetiques))


def test_recalibrage_retrouve_les_coefficients_d_origine(donnees_synthetiques):
    """Sur des donnees engendrees par la formule, le recalibrage doit la retrouver.

    C'est le test qui garantit que l'ajustement est correct : si la cible suit
    exactement la formule de Stull, les coefficients estimes doivent revenir aux
    coefficients publies, et l'erreur au niveau du bruit injecte.
    """
    modele = StullRecalibre().fit(
        donnees_synthetiques[["T2M", "RH2M"]], donnees_synthetiques["T2M_U_med"].to_numpy()
    )
    prediction = modele.predict(donnees_synthetiques)
    rmse = float(np.sqrt(np.mean((prediction - donnees_synthetiques["T2M_U_med"]) ** 2)))
    assert rmse < 0.1
    for estime, publie in zip(modele.coeffs_, COEFFS_STULL_ORIGINE):
        assert estime == pytest.approx(publie, abs=0.01)


def test_recalibrage_bat_la_formule_brute_quand_le_capteur_est_decale(donnees_synthetiques):
    """Si les capteurs lisent 1,5 degre plus bas, le recalibrage doit absorber l'ecart.

    C'est exactement la situation du projet : la formule de Stull est etablie
    pour un abri a 2 m, alors que la mesure de reference vient d'un capteur sans
    abri a 50 cm.
    """
    decale = donnees_synthetiques.assign(T2M_U_med=donnees_synthetiques["T2M_U_med"] - 1.5)
    cible = decale["T2M_U_med"].to_numpy()

    brut = float(np.sqrt(np.mean((StullBrut().predict(decale) - cible) ** 2)))
    recalibre_modele = StullRecalibre().fit(decale[["T2M", "RH2M"]], cible)
    recalibre = float(np.sqrt(np.mean((recalibre_modele.predict(decale) - cible) ** 2)))

    assert brut == pytest.approx(1.5, abs=0.1)
    assert recalibre < 0.2


def test_formule_recalibree_est_lisible(donnees_synthetiques):
    modele = StullRecalibre().fit(
        donnees_synthetiques[["T2M", "RH2M"]], donnees_synthetiques["T2M_U_med"].to_numpy()
    )
    texte = modele.formule()
    assert texte.startswith("Tw =")
    assert "hr^" in texte and "Ts" in texte


def test_polynomiale_degre_1_retrouve_une_relation_lineaire():
    generateur = np.random.default_rng(3)
    x1 = generateur.uniform(-5, 5, 2000)
    x2 = generateur.uniform(0, 100, 2000)
    y = 2.0 * x1 - 0.03 * x2 + 1.0
    df = pd.DataFrame({"a": x1, "b": x2})
    modele = construire_polynomiale(1, 0.01).fit(df, y)
    assert float(np.sqrt(np.mean((modele.predict(df) - y) ** 2))) < 0.05


def test_selection_polynomiale_prefere_le_jeu_de_variables_utile():
    """La selection doit retenir les variables qui portent le signal.

    La cible depend de `utile` seulement : un jeu de variables qui l'ignore doit
    etre ecarte au profit de celui qui la contient.
    """
    generateur = np.random.default_rng(11)
    taille = 4000
    frame = pd.DataFrame(
        {
            "utile": generateur.uniform(-5, 5, taille),
            "bruit": generateur.normal(0, 1, taille),
        }
    )
    frame["cible"] = 1.5 * frame["utile"] ** 2 - 2 * frame["utile"] + generateur.normal(0, 0.1, taille)
    train, validation = frame.iloc[:3000], frame.iloc[3000:]

    modele, grille = selectionner_polynomiale(
        train=train,
        validation=validation,
        cible="cible",
        jeux_variables={"inutile": ["bruit"], "pertinent": ["utile", "bruit"]},
        degres=[1, 2],
        alphas=[0.01, 1.0],
    )
    assert modele.details["jeu_variables"] == "pertinent"
    assert modele.details["degre"] == 2
    assert grille["rmse_validation"].is_monotonic_increasing
