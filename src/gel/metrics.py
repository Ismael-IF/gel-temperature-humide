"""Métriques de régression, de détection du gel et de cout de décision.

Trois niveaux de lecture, volontairement séparés :

1. l'erreur en degrés, qui intéresse l'équipe technique ;
2. la qualité de détection du gel (la décision réellement prise par le
   produit), qui intéresse l'agronome ;
3. le cout des erreurs, qui intéresse celui qui paie l'abonnement et arbitre
   entre une intervention inutile et une parcelle perdue.

Un modèle peut gagner sur le premier niveau et perdre sur les deux autres :
c'est pour cela que les trois sont calculés systématiquement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Les métriques de détection d'évènements rares portent des noms consacrés en
# météorologie : POD (probability of detection), FAR (false alarm ratio),
# CSI (critical success index). Ils sont repris ici pour pouvoir discuter les
# résultats avec des prévisionnistes dans leur propre vocabulaire.


def metriques_regression(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Erreur en degrés, sous plusieurs angles complémentaires.

    Le biais est signé : un modèle qui surestime systématiquement la
    température humide sous-estime le risque de gel, ce qui est l'erreur la
    plus couteuse pour l'exploitant. La RMSE seule le masque complètement.
    """
    y_vrai = np.asarray(y_vrai, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    erreur = y_pred - y_vrai
    erreur_abs = np.abs(erreur)
    variance_cible = np.var(y_vrai)
    return {
        "n": int(y_vrai.size),
        "rmse": float(np.sqrt(np.mean(erreur**2))),
        "mae": float(np.mean(erreur_abs)),
        "biais": float(np.mean(erreur)),
        "ecart_type_erreur": float(np.std(erreur)),
        "erreur_abs_p90": float(np.percentile(erreur_abs, 90)),
        "erreur_abs_p95": float(np.percentile(erreur_abs, 95)),
        "erreur_abs_max": float(erreur_abs.max()),
        "r2": float(1 - np.mean(erreur**2) / variance_cible) if variance_cible > 0 else float("nan"),
    }


def taux_depassement(
    y_vrai: np.ndarray, y_pred: np.ndarray, seuils: tuple[float, ...] = (1.0, 2.0)
) -> dict[str, float]:
    """Fréquence de dépassement des seuils d'erreur, en pourcentage.

    Demandé explicitement par l'énoncé. C'est une lecture plus parlante que la
    RMSE pour un utilisateur final : « dans quelle proportion des heures le
    modèle se trompe-t-il de plus de 1 degré ? »
    """
    erreur_abs = np.abs(np.asarray(y_pred, dtype="float64") - np.asarray(y_vrai, dtype="float64"))
    return {
        f"depassement_{seuil:g}C_pct": float(100 * np.mean(erreur_abs > seuil)) for seuil in seuils
    }


def matrice_gel(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    seuil_observe: float = 0.0,
    seuil_alerte: float | None = None,
) -> dict[str, float]:
    """Table de contingence du classement gel / pas gel.

    L'évènement observé est « température humide sous le seuil agronomique ».
    Le seuil d'alerte du modèle peut être différent du seuil observé : décaler
    l'alerte vers le haut (par exemple alerter dès 0,5 degré) revient a accepter
    des fausses alertes pour rater moins de gelées. C'est le levier de réglage
    produit, étudié par `balayage_seuil_alerte`.
    """
    seuil_alerte = seuil_observe if seuil_alerte is None else seuil_alerte
    observe = np.asarray(y_vrai, dtype="float64") < seuil_observe
    alerte = np.asarray(y_pred, dtype="float64") < seuil_alerte

    vp = int(np.sum(observe & alerte))
    fp = int(np.sum(~observe & alerte))
    fn = int(np.sum(observe & ~alerte))
    vn = int(np.sum(~observe & ~alerte))

    pod = vp / (vp + fn) if (vp + fn) else float("nan")
    far = fp / (vp + fp) if (vp + fp) else float("nan")
    precision = 1 - far if (vp + fp) else float("nan")
    csi = vp / (vp + fp + fn) if (vp + fp + fn) else float("nan")
    f1 = 2 * vp / (2 * vp + fp + fn) if (2 * vp + fp + fn) else float("nan")
    return {
        "seuil_alerte": float(seuil_alerte),
        "vrais_positifs": vp,
        "faux_positifs": fp,
        "faux_negatifs": fn,
        "vrais_negatifs": vn,
        "frequence_gel_observe_pct": float(100 * observe.mean()),
        "pod_rappel": float(pod),
        "far_fausses_alertes": float(far),
        "precision": float(precision),
        "csi": float(csi),
        "f1": float(f1),
        "exactitude": float((vp + vn) / observe.size),
        "biais_frequence": float((vp + fp) / (vp + fn)) if (vp + fn) else float("nan"),
    }


def cout_decision(
    contingence: dict[str, float],
    cout_gelee_non_detectee: float,
    cout_fausse_alerte: float,
    surface_ha: float = 1.0,
) -> dict[str, float]:
    """Cout des erreurs de décision, sous hypothèses économiques explicites.

    Le cout d'une gelée non détectée (perte de récolte) et celui d'une fausse
    alerte (mobilisation inutile : aspersion, tours antigel, main-d'oeuvre) sont
    très asymétriques. Les deux valeurs sont des paramètres de configuration, a
    rediscuter culture par culture : ce calcul donne un ordre de grandeur et un
    cadre d'arbitrage, pas une vérité comptable.
    """
    fn = contingence["faux_negatifs"]
    fp = contingence["faux_positifs"]
    cout_total = surface_ha * (fn * cout_gelee_non_detectee + fp * cout_fausse_alerte)
    return {
        "cout_total_eur": float(cout_total),
        "cout_gelees_ratees_eur": float(surface_ha * fn * cout_gelee_non_detectee),
        "cout_fausses_alertes_eur": float(surface_ha * fp * cout_fausse_alerte),
        "part_cout_gelees_ratees_pct": float(
            100 * fn * cout_gelee_non_detectee / (fn * cout_gelee_non_detectee + fp * cout_fausse_alerte)
        )
        if (fn or fp)
        else float("nan"),
    }


def balayage_seuil_alerte(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    seuils_alerte: np.ndarray,
    cout_gelee_non_detectee: float,
    cout_fausse_alerte: float,
    seuil_observe: float = 0.0,
    surface_ha: float = 1.0,
) -> pd.DataFrame:
    """Cout et qualité de détection en fonction du seuil de déclenchement d'alerte.

    Résultat : la courbe de cout en fonction du seuil, dont le minimum donne le
    réglage recommandé pour le couple de couts fourni. C'est la traduction
    opérationnelle d'un modèle de régression en règle de décision.
    """
    lignes = []
    for seuil in seuils_alerte:
        contingence = matrice_gel(y_vrai, y_pred, seuil_observe, float(seuil))
        cout = cout_decision(
            contingence, cout_gelee_non_detectee, cout_fausse_alerte, surface_ha
        )
        lignes.append({**contingence, **cout})
    return pd.DataFrame(lignes)


def tableau_metriques(
    y_vrai: np.ndarray,
    y_pred: np.ndarray,
    seuils_erreur: tuple[float, ...] = (1.0, 2.0),
    seuil_gel: float = 0.0,
) -> dict[str, float]:
    """Regroupe erreur, dépassements et détection du gel en une seule ligne."""
    resultat = metriques_regression(y_vrai, y_pred)
    resultat.update(taux_depassement(y_vrai, y_pred, seuils_erreur))
    contingence = matrice_gel(y_vrai, y_pred, seuil_gel)
    resultat.update(
        {
            "pod_rappel_gel": contingence["pod_rappel"],
            "far_gel": contingence["far_fausses_alertes"],
            "csi_gel": contingence["csi"],
            "gelees_ratees": contingence["faux_negatifs"],
            "fausses_alertes": contingence["faux_positifs"],
        }
    )
    return resultat


def agreger_par_nuit(
    df: pd.DataFrame,
    colonne_horodatage: str = "fin_fenetre",
    colonne_station: str = "station_id",
    colonne_cible: str = "T2M_U_med",
    colonne_prediction: str = "prediction",
    decalage_heures: int = 12,
) -> pd.DataFrame:
    """Agrège les heures en évènements « nuit de gel », par station.

    C'est l'unité de décision réelle, et elle change les conclusions
    économiques. Une gelée est un évènement nocturne : si une nuit compte huit
    heures sous zéro, l'exploitant ne perd pas huit fois sa récolte, et il ne
    déclenche pas huit fois son dispositif antigel. Compter le cout a l'heure
    multiplie donc artificiellement les pertes par la durée de l'épisode.

    Une nuit est définie de midi a midi (`decalage_heures`), ce qui garde dans
    le même évènement la soirée et la fin de nuit qui la suit. L'évènement est
    résumé par le minimum de la nuit : c'est le moment le plus froid qui décide
    du dégat, et c'est aussi sur un minimum annoncé que se déclenche une alerte.
    """
    nuit = (df[colonne_horodatage] - pd.Timedelta(hours=decalage_heures)).dt.normalize()
    agrege = (
        df.assign(nuit=nuit)
        .groupby([colonne_station, "nuit"], observed=True)
        .agg(
            minimum_mesure=(colonne_cible, "min"),
            minimum_prediction=(colonne_prediction, "min"),
            heures=(colonne_cible, "size"),
        )
        .reset_index()
    )
    return agrege


def comparaison_par_nuit(
    y_vrai: np.ndarray,
    predictions: dict[str, np.ndarray],
    nuits: np.ndarray,
    reference: str,
) -> pd.DataFrame:
    """Compare les modèles nuit par nuit, au lieu d'un seul chiffre global.

    Deux modèles séparés par 0,01 degré de RMSE ne sont pas départagés, et il
    faut pouvoir le dire. Plutot que de comparer deux nombres, on calcule la
    RMSE de chaque modèle **sur chaque nuit** de la période de test, puis on
    regarde deux choses :

    - combien de nuits chaque modèle gagne, ce qui dit si un modèle domine ou
      si le classement change d'une nuit à l'autre ;
    - l'écart moyen de RMSE par rapport au modèle de référence, et son écart
      type d'une nuit à l'autre. Si l'écart moyen est dix fois plus petit que
      sa propre dispersion, l'écart global ne veut rien dire.

    La nuit est la bonne maille de regroupement : les heures d'une même nuit se
    ressemblent beaucoup, les traiter comme des observations indépendantes
    donnerait une fausse impression de précision.
    """
    erreurs = pd.DataFrame({nom: (pred - y_vrai) ** 2 for nom, pred in predictions.items()})
    erreurs["nuit"] = nuits
    rmse_par_nuit = erreurs.groupby("nuit").mean() ** 0.5

    meilleur_par_nuit = rmse_par_nuit.idxmin(axis=1).value_counts()
    lignes = []
    for nom in predictions:
        ecart = rmse_par_nuit[nom] - rmse_par_nuit[reference]
        lignes.append(
            {
                "modele": nom,
                "rmse_globale": float(np.sqrt(np.mean((predictions[nom] - y_vrai) ** 2))),
                "rmse_mediane_par_nuit": float(rmse_par_nuit[nom].median()),
                "nuits_gagnees": int(meilleur_par_nuit.get(nom, 0)),
                "nuits_evaluees": int(len(rmse_par_nuit)),
                "ecart_moyen_vs_reference": float(ecart.mean()),
                "ecart_type_de_l_ecart": float(ecart.std()),
            }
        )
    return pd.DataFrame(lignes).sort_values("rmse_globale").reset_index(drop=True)


def metriques_par_groupe(
    df: pd.DataFrame,
    colonne_groupe: str,
    colonne_cible: str,
    colonne_prediction: str,
    minimum_observations: int = 50,
) -> pd.DataFrame:
    """Métriques déclinées par groupe (mois, heure, station, classe de température).

    Une RMSE globale cache les régimes ou le modèle échoue. Sur un produit
    d'alerte gel, la seule performance qui compte est celle des nuits froides :
    ce découpage sert a vérifier que la performance moyenne n'est pas portée par
    les situations sans enjeu.
    """
    lignes = []
    for valeur, groupe in df.groupby(colonne_groupe, observed=True):
        if len(groupe) < minimum_observations:
            continue
        metriques = metriques_regression(groupe[colonne_cible], groupe[colonne_prediction])
        lignes.append({colonne_groupe: valeur, **metriques})

    resultat = pd.DataFrame(lignes)
    # Un groupe categoriel (les regimes de temperature) doit garder son ordre
    # logique : trie en chaines de caracteres, « < -3 » viendrait apres « 0 a 2 ».
    if isinstance(df[colonne_groupe].dtype, pd.CategoricalDtype):
        resultat[colonne_groupe] = pd.Categorical(
            resultat[colonne_groupe],
            categories=df[colonne_groupe].cat.categories,
            ordered=True,
        )
    return resultat.sort_values(colonne_groupe).reset_index(drop=True)
