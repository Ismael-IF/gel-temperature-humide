"""Les trois familles de modèles comparées, plus les références de contrôle.

L'énoncé demande trois approches : la formule de Stull, une régression
polynomiale et un modèle d'apprentissage automatique. On y ajoute deux
références qui ne couteraient rien a un industriel et qu'il faut battre pour
justifier un modèle :

- `Persistance2m` : prédire la température humide par la température sous abri.
- `StullRecalibre` : la même formule physique, mais dont les quatre
  coefficients sont réestimés sur nos propres capteurs. Elle départage deux
  explications possibles d'un mauvais score de la formule d'origine : la forme
  fonctionnelle est inadaptée, ou bien elle est correcte mais calibrée pour un
  abri a 2 m alors que nos capteurs mesurent a 50 cm sans abri.

Toutes les classes suivent l'interface scikit-learn (`fit` / `predict`) et
acceptent un DataFrame, ce qui évite les erreurs d'ordre de colonnes entre
l'apprentissage et la production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .features import (
    COEFFS_STULL_ORIGINE,
    EXPOSANT_STULL_ORIGINE,
    matrice_stull,
    temperature_humide_stull,
)


@dataclass
class ModeleEvalue:
    """Un modèle entrainé, avec la liste exacte des colonnes qu'il consomme."""

    nom: str
    estimateur: Any
    colonnes: list[str]
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def predire(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimateur.predict(df[self.colonnes]), dtype="float64")


class Persistance2m(BaseEstimator, RegressorMixin):
    """Référence triviale : la température humide vaut la température sous abri.

    Ce modèle ne nécessite ni données ni calcul. Tout modèle qui ne le bat pas
    largement ne justifie pas sa mise en production.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "Persistance2m":
        self.colonne_ = "T2M"
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X["T2M"], dtype="float64")


class StullBrut(BaseEstimator, RegressorMixin):
    """Formule de Stull simplifiée, coefficients publiés, aucun apprentissage.

    C'est la seule des trois approches qui n'a vu aucune de nos données. Elle
    donne donc la performance atteignable sans projet data du tout, et sert de
    point zéro a la comparaison.
    """

    def __init__(
        self,
        coeffs: tuple[float, float, float, float] = COEFFS_STULL_ORIGINE,
        exposant: float = EXPOSANT_STULL_ORIGINE,
    ) -> None:
        self.coeffs = coeffs
        self.exposant = exposant

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "StullBrut":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return temperature_humide_stull(X["T2M"], X["RH2M"], self.coeffs, self.exposant)


class StullRecalibre(BaseEstimator, RegressorMixin):
    """Formule de Stull dont les quatre coefficients sont réestimés sur nos mesures.

    L'exposant de l'humidité reste celui publié par Stull (0,2735). Une fois cet
    exposant fixé, la formule

        Tw = a * hr^p * Ts + b * hr^2 + c * hr + d

    est **linéaire** en (a, b, c, d) : la recalibrer est donc une simple
    régression linéaire sur les trois colonnes [hr^p * Ts, hr^2, hr]. Pas
    d'optimisation itérative, pas de point de départ à choisir, pas de risque de
    non-convergence.

    J'ai testé l'alternative, qui consiste à optimiser aussi l'exposant : elle
    fait gagner 0,002 °C de RMSE sur la période de test, pour un paramètre de
    plus et une formule qui s'éloigne de la littérature sans raison physique. Je
    garde donc l'exposant publié.
    """

    def __init__(self, exposant: float = EXPOSANT_STULL_ORIGINE) -> None:
        self.exposant = exposant

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "StullRecalibre":
        matrice = matrice_stull(X["T2M"], X["RH2M"], self.exposant)
        cible = np.asarray(y, dtype="float64")

        coeffs, *_ = np.linalg.lstsq(matrice, cible, rcond=None)
        self.coeffs_ = tuple(float(c) for c in coeffs)
        self.exposant_ = self.exposant
        self.rmse_apprentissage_ = float(np.sqrt(np.mean((matrice @ coeffs - cible) ** 2)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrice = matrice_stull(X["T2M"], X["RH2M"], self.exposant_)
        return matrice @ np.asarray(self.coeffs_, dtype="float64")

    def formule(self) -> str:
        """Écrit la formule recalibrée, pour pouvoir la comparer a l'originale."""
        a, b, c, d = self.coeffs_
        signe = lambda valeur, format_: f"{'-' if valeur < 0 else '+'} {abs(valeur):{format_}}"
        return (
            f"Tw = {a:.4f} * hr^{self.exposant_:.4f} * Ts "
            f"{signe(b, '.3e')} * hr^2 "
            f"{signe(c, '.5f')} * hr "
            f"{signe(d, '.4f')}"
        )


def construire_polynomiale(degre: int, alpha: float) -> Pipeline:
    """Régression polynomiale régularisée.

    Trois choix a justifier :

    - la standardisation précède l'élévation en puissance, sinon les termes de
      degré 3 de variables d'échelles très différentes (SSI en W/m2, VPD en kPa)
      produisent une matrice mal conditionnée ;
    - la régularisation Ridge est indispensable des le degré 2, car les termes
      croisés sont fortement colinéaires (RH2M, VPD et l'écart au point de rosée
      décrivent la même physique) ;
    - le niveau de régularisation et le degré sont choisis sur une saison de
      validation tenue a l'écart, jamais sur les données de test.
    """
    return Pipeline(
        [
            ("echelle", StandardScaler()),
            ("polynome", PolynomialFeatures(degree=degre, include_bias=False)),
            ("ridge", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def construire_gbm(parametres: dict[str, Any], graine: int = 42) -> HistGradientBoostingRegressor:
    """Gradient boosting sur histogrammes.

    Choisi plutot qu'une foret aléatoire ou un réseau de neurones pour trois
    raisons tenant au contexte : il encaisse le million et demi de lignes en
    quelques dizaines de secondes sans sous-échantillonnage, il ne demande ni
    mise a l'échelle ni imputation, et son empreinte mémoire reste compatible
    avec un réentrainement périodique sur une machine ordinaire. Une foret
    aléatoire a 50 arbres non élagués sur ce volume dépasse plusieurs gigaoctets
    de modèle sérialisé, pour une performance comparable.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=graine,
        early_stopping=False,
        **parametres,
    )


def selectionner_polynomiale(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cible: str,
    jeux_variables: dict[str, Sequence[str]],
    degres: Sequence[int],
    alphas: Sequence[float],
    max_termes: int = 400,
    train_final: pd.DataFrame | None = None,
) -> tuple[ModeleEvalue, pd.DataFrame]:
    """Choisit jeu de variables, degré et régularisation sur la saison de validation.

    C'est ce que veut dire « la régression polynomiale la plus pertinente
    possible » : non pas un degré choisi d'avance, mais le meilleur compromis
    mesuré sur des données que le modèle n'a pas vues. La borne `max_termes`
    écarte les combinaisons dont le développement polynomial explose, qui ne
    tiennent pas en mémoire et qui ne sont de toute façon pas déployables.

    `train_final` permet de réajuster la configuration retenue sur apprentissage
    + validation : une fois les hyperparamètres choisis, se priver de la saison
    la plus récente n'aurait pas de sens pour prédire la saison suivante.
    """
    resultats = []
    for nom_jeu, colonnes in jeux_variables.items():
        colonnes = list(colonnes)
        for degre in degres:
            nombre_termes = PolynomialFeatures(degree=degre, include_bias=False).fit(
                np.zeros((1, len(colonnes)))
            ).n_output_features_
            if nombre_termes > max_termes:
                continue
            for alpha in alphas:
                modele = construire_polynomiale(degre, alpha)
                modele.fit(train[colonnes], train[cible])
                prediction = modele.predict(validation[colonnes])
                rmse = float(
                    np.sqrt(np.mean((prediction - validation[cible].to_numpy()) ** 2))
                )
                resultats.append(
                    {
                        "jeu_variables": nom_jeu,
                        "degre": degre,
                        "alpha": alpha,
                        "nb_termes": int(nombre_termes),
                        "rmse_validation": rmse,
                    }
                )

    grille = pd.DataFrame(resultats).sort_values("rmse_validation").reset_index(drop=True)
    meilleur = grille.iloc[0]
    colonnes = list(jeux_variables[meilleur["jeu_variables"]])
    donnees_finales = train if train_final is None else train_final
    estimateur = construire_polynomiale(int(meilleur["degre"]), float(meilleur["alpha"]))
    estimateur.fit(donnees_finales[colonnes], donnees_finales[cible])
    modele = ModeleEvalue(
        nom="Régression polynomiale",
        estimateur=estimateur,
        colonnes=colonnes,
        description=(
            f"Polynome de degré {int(meilleur['degre'])} sur le jeu "
            f"« {meilleur['jeu_variables']} » ({int(meilleur['nb_termes'])} termes), "
            f"Ridge alpha={meilleur['alpha']}"
        ),
        details=meilleur.to_dict(),
    )
    return modele, grille


def selectionner_gbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cible: str,
    colonnes: Sequence[str],
    grille: Sequence[dict[str, Any]],
    graine: int = 42,
) -> tuple[ModeleEvalue, pd.DataFrame]:
    """Choisit les hyperparamètres du gradient boosting sur la saison de validation."""
    colonnes = list(colonnes)
    resultats = []
    meilleur_estimateur = None
    meilleure_rmse = np.inf
    meilleure_config: dict[str, Any] = {}

    for parametres in grille:
        estimateur = construire_gbm(dict(parametres), graine)
        estimateur.fit(train[colonnes], train[cible])
        prediction = estimateur.predict(validation[colonnes])
        rmse = float(np.sqrt(np.mean((prediction - validation[cible].to_numpy()) ** 2)))
        resultats.append({**parametres, "rmse_validation": rmse})
        if rmse < meilleure_rmse:
            meilleure_rmse, meilleur_estimateur, meilleure_config = rmse, estimateur, dict(parametres)

    # Le modèle retenu est réentrainé sur apprentissage + validation : une fois
    # les hyperparamètres choisis, se priver de la saison la plus récente serait
    # absurde pour un modèle qui doit prédire la saison suivante.
    complet = pd.concat([train, validation], ignore_index=True)
    estimateur_final = construire_gbm(meilleure_config, graine)
    estimateur_final.fit(complet[colonnes], complet[cible])

    modele = ModeleEvalue(
        nom="Gradient boosting",
        estimateur=estimateur_final,
        colonnes=colonnes,
        description=(
            "HistGradientBoostingRegressor, hyperparamètres choisis sur la saison "
            f"de validation : {meilleure_config}"
        ),
        details={**meilleure_config, "rmse_validation": meilleure_rmse},
    )
    return modele, pd.DataFrame(resultats).sort_values("rmse_validation").reset_index(drop=True)
