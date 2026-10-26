"""Enchainement complet : nettoyage, apprentissage, évaluation, figures.

Chaque étape écrit ses sorties sur disque et peut être relancée seule, ce qui
permet de refaire la modélisation sans refaire le nettoyage.

Tous les chiffres qui apparaissent dans `rapports/resultats.md` sortent des
fichiers écrits ici, dans `artefacts/` : chaque tableau du rapport a son CSV, et
chaque exécution les réécrit. Si un chiffre du rapport semble douteux, le CSV
correspondant permet de le retrouver.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .cleaning import ajouter_horodatage, nettoyer
from .config import Config
from .features import construire_features, jeux_de_variables
from .io import ecrire_parquet, lire_parquet
from .metrics import (
    agreger_par_nuit,
    balayage_seuil_alerte,
    comparaison_par_nuit,
    cout_decision,
    matrice_gel,
    metriques_par_groupe,
    tableau_metriques,
)
from .models import (
    ModeleEvalue,
    Persistance2m,
    StullBrut,
    StullRecalibre,
    selectionner_gbm,
    selectionner_polynomiale,
)

LOGGER = logging.getLogger(__name__)

# Échantillon utilisé pour la seule recherche d'hyperparamètres polynomiaux.
# Le développement d'un polynome de degré 3 sur 10 variables produit 285 termes :
# a 920 000 lignes la matrice dépasse 2 Go, pour un classement des configurations
# qui ne change pas. Le modèle retenu est, lui, réajusté sur la totalité.
TAILLE_ECHANTILLON_POLY = 250_000


@dataclass
class Decoupage:
    """Les trois jeux de données, séparés dans le temps."""

    apprentissage: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def apprentissage_complet(self) -> pd.DataFrame:
        return pd.concat([self.apprentissage, self.validation], ignore_index=True)

    def resume(self) -> pd.DataFrame:
        lignes = []
        for nom, jeu in (
            ("apprentissage", self.apprentissage),
            ("validation", self.validation),
            ("test", self.test),
        ):
            lignes.append(
                {
                    "jeu": nom,
                    "lignes": len(jeu),
                    "debut": jeu["date"].min().date().isoformat(),
                    "fin": jeu["date"].max().date().isoformat(),
                    "stations": int(jeu["station_id"].nunique()),
                    "part_gel_pct": round(100 * float((jeu["T2M_U_med"] < 0).mean()), 2),
                    "heures_de_gel": int((jeu["T2M_U_med"] < 0).sum()),
                }
            )
        return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# Étape 1 : nettoyage
# --------------------------------------------------------------------------
def etape_nettoyage(cfg: Config) -> pd.DataFrame:
    """Applique les filtres de l'énoncé et écrit le Parquet nettoyé."""
    depart = time.perf_counter()
    brut = lire_parquet(cfg.chemin("raw_parquet"))
    LOGGER.info("donnees brutes : %d lignes", len(brut))

    propre, journal = nettoyer(brut, cfg.nettoyage)
    propre = ajouter_horodatage(propre)

    ecrire_parquet(propre, cfg.chemin("clean_parquet"))
    journal.to_csv(cfg.chemin("audit_csv"), index=False, encoding="utf-8")

    viz.figure_journal_nettoyage(journal, cfg.chemin("figures_dir") / "nettoyage_volumes.png")

    resume = {
        "lignes_brutes": int(journal.attrs["lignes_initiales"]),
        "lignes_nettoyees": int(len(propre)),
        "part_conservee_pct": round(100 * len(propre) / journal.attrs["lignes_initiales"], 2),
        "stations_brutes": int(brut["station_id"].nunique()),
        "stations_conservees": int(propre["station_id"].nunique()),
        "periode": [propre["date"].min().date().isoformat(), propre["date"].max().date().isoformat()],
        "part_gel_pct": round(100 * float((propre["T2M_U_med"] < 0).mean()), 2),
        "duree_s": round(time.perf_counter() - depart, 1),
    }
    _ecrire_json(cfg.chemin("artefacts_dir") / "resume_nettoyage.json", resume)
    LOGGER.info("nettoyage termine : %s", resume)
    return propre


# --------------------------------------------------------------------------
# Étape 2 : découpage temporel et variables
# --------------------------------------------------------------------------
def preparer_decoupage(cfg: Config) -> Decoupage:
    """Charge les données nettoyées, construit les variables et découpe dans le temps.

    Le découpage est strictement chronologique, comme l'impose l'énoncé :
    apprentissage du 15/03/2020 au 14/03/2023, test a partir du 15/03/2023. La
    dernière saison de l'apprentissage sert de validation pour choisir les
    modèles. Un découpage aléatoire serait ici un piège : les stations voisines
    partagent les mêmes variables atmosphériques estimées a la même heure, donc
    des lignes quasi identiques se retrouveraient des deux cotés et la
    performance mesurée serait surévaluée. L'expérience
    `experience_protocole_evaluation` chiffre cet écart.
    """
    propre = lire_parquet(cfg.chemin("clean_parquet"))
    propre = construire_features(propre)
    propre["mois"] = propre["date"].dt.month.astype("int8")

    debut_train = pd.Timestamp(cfg.date("train_debut"))
    fin_train = pd.Timestamp(cfg.date("train_fin"))
    debut_test = pd.Timestamp(cfg.date("test_debut"))
    debut_validation = pd.Timestamp(cfg.date("validation_debut"))

    fenetre_train = (propre["date"] >= debut_train) & (propre["date"] <= fin_train)
    apprentissage = propre.loc[fenetre_train & (propre["date"] < debut_validation)]
    validation = propre.loc[fenetre_train & (propre["date"] >= debut_validation)]
    test = propre.loc[propre["date"] >= debut_test]

    return Decoupage(
        apprentissage=apprentissage.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Étape 3 : apprentissage des trois approches
# --------------------------------------------------------------------------
def entrainer_modeles(cfg: Config, decoupage: Decoupage) -> tuple[list[ModeleEvalue], dict]:
    """Entraine les cinq modèles comparés et renvoie les traces de sélection."""
    cible = cfg.cible
    jeux = jeux_de_variables()
    graine = cfg.graine
    traces: dict = {}

    apprentissage = decoupage.apprentissage
    validation = decoupage.validation
    complet = decoupage.apprentissage_complet

    modeles: list[ModeleEvalue] = [
        ModeleEvalue(
            nom="Persistance 2 m",
            estimateur=Persistance2m().fit(apprentissage),
            colonnes=["T2M"],
            description="Référence sans modèle : la température humide vaut la température sous abri",
        ),
        ModeleEvalue(
            nom="Stull (coefficients publiés)",
            estimateur=StullBrut(),
            colonnes=["T2M", "RH2M"],
            description="Formule de Stull simplifiée, coefficients d'origine, aucun apprentissage",
        ),
    ]

    # Stull recalibré sur nos capteurs (apprentissage + validation).
    depart = time.perf_counter()
    stull_recalibre = StullRecalibre()
    stull_recalibre.fit(complet[["T2M", "RH2M"]], complet[cible].to_numpy())
    modeles.append(
        ModeleEvalue(
            nom="Stull recalibré",
            estimateur=stull_recalibre,
            colonnes=["T2M", "RH2M"],
            description="Formule de Stull dont les quatre coefficients et l'exposant sont réestimés sur nos mesures",
            details={
                "formule": stull_recalibre.formule(),
                "coeffs": list(stull_recalibre.coeffs_),
                "exposant": stull_recalibre.exposant_,
            },
        )
    )
    traces["stull_recalibre"] = {
        "formule_origine": "Tw = 0.2831 * hr^0.2735 * Ts + 3.018e-04 * hr^2 + 0.01289 * hr - 4.0962",
        "formule_recalibree": stull_recalibre.formule(),
        "duree_s": round(time.perf_counter() - depart, 1),
    }

    # Régression polynomiale : sélection sur échantillon, ajustement final complet.
    depart = time.perf_counter()
    echantillon = apprentissage.sample(
        n=min(TAILLE_ECHANTILLON_POLY, len(apprentissage)), random_state=graine
    )
    modele_poly, grille_poly = selectionner_polynomiale(
        train=_en_float32(echantillon, jeux, cible),
        validation=_en_float32(validation, jeux, cible),
        cible=cible,
        jeux_variables=jeux,
        degres=cfg.modeles["poly_degres"],
        alphas=cfg.modeles["poly_alphas"],
        train_final=_en_float32(complet, jeux, cible),
    )
    modeles.append(modele_poly)
    grille_poly.to_csv(cfg.chemin("artefacts_dir") / "grille_polynomiale.csv", index=False)
    traces["polynomiale"] = {
        "configuration_retenue": modele_poly.details,
        "lignes_selection": int(len(echantillon)),
        "duree_s": round(time.perf_counter() - depart, 1),
    }

    # Gradient boosting sur le jeu de variables complet.
    depart = time.perf_counter()
    modele_gbm, grille_gbm = selectionner_gbm(
        train=apprentissage,
        validation=validation,
        cible=cible,
        colonnes=jeux["complet"],
        grille=cfg.modeles["gbm_grille"],
        graine=graine,
    )
    modeles.append(modele_gbm)
    grille_gbm.to_csv(cfg.chemin("artefacts_dir") / "grille_gbm.csv", index=False)
    traces["gradient_boosting"] = {
        "configuration_retenue": modele_gbm.details,
        "duree_s": round(time.perf_counter() - depart, 1),
    }

    return modeles, traces


def _en_float32(df: pd.DataFrame, jeux: dict[str, list[str]], cible: str) -> pd.DataFrame:
    """Réduit la précision des colonnes utilisées par la régression polynomiale.

    Le développement polynomial double la mémoire a chaque passage en float64 :
    rester en float32 divise par deux l'empreinte sans effet mesurable sur des
    mesures connues au dixième de degré.
    """
    colonnes = sorted({c for liste in jeux.values() for c in liste} | {cible})
    return df[colonnes].astype("float32")


# --------------------------------------------------------------------------
# Étape 4 : évaluation
# --------------------------------------------------------------------------
def evaluer(cfg: Config, decoupage: Decoupage, modeles: list[ModeleEvalue]) -> dict:
    """Évalue tous les modèles sur la période de test et produit tables et figures."""
    cible = cfg.cible
    test = decoupage.test
    y_test = test[cible].to_numpy(dtype="float64")
    seuils_erreur = tuple(cfg.evaluation["seuils_erreur"])
    seuil_gel = float(cfg.evaluation["seuil_gel"])
    figures = cfg.chemin("figures_dir")
    artefacts = cfg.chemin("artefacts_dir")

    predictions = {modele.nom: modele.predire(test) for modele in modeles}

    # Les predictions de test sont conservees : elles permettent de refaire toute
    # l'analyse (metriques, agregation par nuit, couts) sans reentrainer, et c'est
    # ce que fait le notebook de demonstration.
    ecrire_parquet(
        pd.DataFrame(
            {
                "station_id": test["station_id"].to_numpy(),
                "fin_fenetre": test["fin_fenetre"].to_numpy(),
                "date": test["date"].to_numpy(),
                "mois": test["mois"].to_numpy(),
                "hour": test["hour"].to_numpy(),
                "T2M": test["T2M"].to_numpy(),
                cible: y_test,
                **{nom: prediction for nom, prediction in predictions.items()},
            }
        ),
        cfg.chemin("predictions"),
    )

    # --- tableau principal ------------------------------------------------
    lignes = []
    for nom, prediction in predictions.items():
        ligne = {"modele": nom}
        ligne.update(tableau_metriques(y_test, prediction, seuils_erreur, seuil_gel))
        contingence = matrice_gel(y_test, prediction, seuil_gel)
        ligne.update(
            cout_decision(
                contingence,
                cfg.metier["cout_gelee_non_detectee_eur_ha"],
                cfg.metier["cout_fausse_alerte_eur_ha"],
                cfg.metier["surface_suivie_ha"],
            )
        )
        lignes.append(ligne)
    tableau = pd.DataFrame(lignes).sort_values("rmse").reset_index(drop=True)
    tableau.to_csv(artefacts / "metriques_test.csv", index=False)

    # --- comparaison des modeles nuit par nuit ----------------------------
    nuits = (test["fin_fenetre"] - pd.Timedelta(hours=12)).dt.normalize()
    reference = tableau.loc[0, "modele"]
    comparaison = comparaison_par_nuit(y_test, predictions, nuits.to_numpy(), reference=reference)
    comparaison.to_csv(artefacts / "comparaison_par_nuit.csv", index=False)

    # --- contingences horaires --------------------------------------------
    contingences = {
        nom: matrice_gel(y_test, prediction, seuil_gel) for nom, prediction in predictions.items()
    }
    pd.DataFrame(contingences).T.reset_index(names="modele").to_csv(
        artefacts / "contingence_gel_horaire.csv", index=False
    )

    # --- contingences et couts a l'echelle de l'evenement -----------------
    # Une gelee est un evenement nocturne : c'est a cette maille que se prend la
    # decision d'intervenir, et donc a cette maille que se chiffre le cout.
    seuils_alerte = np.round(np.arange(-1.0, 3.01, 0.1), 2)
    contingences_nuit, balayages, evenements = {}, {}, {}
    for nom, prediction in predictions.items():
        agrege = agreger_par_nuit(
            test[["station_id", "fin_fenetre", cible]].assign(prediction=prediction),
            colonne_cible=cible,
        )
        evenements[nom] = agrege
        mesure_nuit = agrege["minimum_mesure"].to_numpy(dtype="float64")
        prediction_nuit = agrege["minimum_prediction"].to_numpy(dtype="float64")
        contingences_nuit[nom] = matrice_gel(mesure_nuit, prediction_nuit, seuil_gel)
        balayages[nom] = balayage_seuil_alerte(
            mesure_nuit,
            prediction_nuit,
            seuils_alerte,
            cfg.metier["cout_gelee_non_detectee_eur_ha"],
            cfg.metier["cout_fausse_alerte_eur_ha"],
            seuil_gel,
            cfg.metier["surface_suivie_ha"],
        )

    table_nuit = pd.DataFrame(contingences_nuit).T.reset_index(names="modele")
    for nom in contingences_nuit:
        cout = cout_decision(
            contingences_nuit[nom],
            cfg.metier["cout_gelee_non_detectee_eur_ha"],
            cfg.metier["cout_fausse_alerte_eur_ha"],
            cfg.metier["surface_suivie_ha"],
        )
        for cle, valeur in cout.items():
            table_nuit.loc[table_nuit["modele"] == nom, cle] = valeur
    table_nuit.to_csv(artefacts / "contingence_gel_nuit.csv", index=False)

    for nom, table in balayages.items():
        table.insert(0, "modele", nom)
    pd.concat(balayages.values(), ignore_index=True).to_csv(
        artefacts / "balayage_seuil_alerte.csv", index=False
    )

    # --- déclinaisons par groupe ------------------------------------------
    detail = test[["mois", "hour", "station_id", cible]].copy()
    bornes = [-np.inf, -3, -1, 0, 2, 5, np.inf]
    etiquettes = ["< -3", "-3 à -1", "-1 à 0", "0 à 2", "2 à 5", "> 5"]
    detail["regime"] = pd.cut(detail[cible], bins=bornes, labels=etiquettes)

    tables_mois, tables_heure, tables_regime, tables_station = {}, {}, {}, {}
    for nom, prediction in predictions.items():
        detail["prediction"] = prediction
        tables_mois[nom] = metriques_par_groupe(detail, "mois", cible, "prediction")
        tables_heure[nom] = metriques_par_groupe(detail, "hour", cible, "prediction")
        tables_regime[nom] = metriques_par_groupe(detail, "regime", cible, "prediction")
        tables_station[nom] = metriques_par_groupe(
            detail, "station_id", cible, "prediction", minimum_observations=100
        )

    for nom_table, tables in (
        ("mois", tables_mois),
        ("heure", tables_heure),
        ("regime", tables_regime),
    ):
        concat = pd.concat(
            [table.assign(modele=nom) for nom, table in tables.items()], ignore_index=True
        )
        concat.to_csv(artefacts / f"metriques_par_{nom_table}.csv", index=False)

    dispersion_stations = pd.DataFrame(
        [
            {
                "modele": nom,
                "stations_evaluees": len(table),
                "rmse_mediane": float(table["rmse"].median()),
                "rmse_p90": float(table["rmse"].quantile(0.9)),
                "rmse_max": float(table["rmse"].max()),
                "stations_rmse_sup_2C": int((table["rmse"] > 2).sum()),
            }
            for nom, table in tables_station.items()
        ]
    )
    dispersion_stations.to_csv(artefacts / "dispersion_par_station.csv", index=False)

    # --- figures ----------------------------------------------------------
    erreurs = {nom: prediction - y_test for nom, prediction in predictions.items()}
    principaux = [
        nom
        for nom in (
            "Stull (coefficients publiés)",
            "Stull recalibré",
            "Régression polynomiale",
            "Gradient boosting",
        )
        if nom in predictions
    ]

    viz.figure_distribution_erreurs(
        {nom: erreurs[nom] for nom in principaux}, figures / "distribution_erreurs.png"
    )
    viz.figure_predictions_vs_mesures(
        {nom: predictions[nom] for nom in principaux}, y_test, figures / "predictions_vs_mesures.png"
    )
    viz.figure_metriques_par_groupe(
        {nom: tables_mois[nom] for nom in principaux},
        "mois",
        figures / "rmse_par_mois.png",
        "Erreur par mois, période de test",
        "mois",
    )
    viz.figure_metriques_par_groupe(
        {nom: tables_heure[nom] for nom in principaux},
        "hour",
        figures / "rmse_par_heure.png",
        "Erreur par heure de la fenêtre agrégée",
        "heure de fin de la fenêtre horaire",
    )
    viz.figure_contingences_gel(
        {nom: contingences_nuit[nom] for nom in principaux}, figures / "contingence_gel.png"
    )
    viz.figure_cout_par_seuil(
        {nom: balayages[nom] for nom in principaux},
        figures / "cout_par_seuil.png",
        cfg.metier["surface_suivie_ha"],
    )
    viz.figure_erreur_par_regime(
        {nom: tables_regime[nom] for nom in principaux}, figures / "erreur_par_regime.png"
    )

    # --- seuil d'alerte recommandé ---------------------------------------
    recommandations = []
    for nom, table in balayages.items():
        meilleur = table.loc[table["cout_total_eur"].idxmin()]
        reference = table.loc[np.isclose(table["seuil_alerte"], seuil_gel)].iloc[0]
        recommandations.append(
            {
                "modele": nom,
                "seuil_optimal": float(meilleur["seuil_alerte"]),
                "cout_au_seuil_optimal_eur": float(meilleur["cout_total_eur"]),
                "cout_au_seuil_0_eur": float(reference["cout_total_eur"]),
                "economie_pct": round(
                    100 * (1 - meilleur["cout_total_eur"] / reference["cout_total_eur"]), 1
                )
                if reference["cout_total_eur"]
                else float("nan"),
                "pod_au_seuil_optimal": float(meilleur["pod_rappel"]),
                "far_au_seuil_optimal": float(meilleur["far_fausses_alertes"]),
            }
        )
    pd.DataFrame(recommandations).to_csv(artefacts / "seuil_alerte_recommande.csv", index=False)

    return {
        "tableau": tableau,
        "contingences_horaires": contingences,
        "contingences_nuit": table_nuit,
        "comparaison_par_nuit": comparaison,
        "predictions": predictions,
        "evenements": evenements,
        "recommandations": pd.DataFrame(recommandations),
        "dispersion_stations": dispersion_stations,
    }


# --------------------------------------------------------------------------
# Étape 5 : expériences complémentaires
# --------------------------------------------------------------------------
def experience_protocole_evaluation(cfg: Config, decoupage: Decoupage) -> dict:
    """Chiffre l'écart entre découpage aléatoire et découpage chronologique.

    Le même modèle est évalué deux fois : une fois sur un tirage aléatoire 70 / 30
    de la période d'apprentissage, une fois sur la saison suivante. L'écart entre
    les deux RMSE est le montant de l'illusion : c'est la performance qu'un
    protocole trop permissif fait croire qu'on a, et qu'on n'a pas en production.
    """
    from sklearn.model_selection import train_test_split

    from .models import construire_gbm

    colonnes = jeux_de_variables()["complet"]
    cible = cfg.cible
    parametres = dict(cfg.modeles["gbm_grille"][0])
    complet = decoupage.apprentissage_complet

    train_alea, test_alea = train_test_split(
        complet, test_size=0.3, random_state=cfg.graine, shuffle=True
    )
    modele_alea = construire_gbm(parametres, cfg.graine).fit(train_alea[colonnes], train_alea[cible])
    rmse_alea = float(
        np.sqrt(
            np.mean((modele_alea.predict(test_alea[colonnes]) - test_alea[cible].to_numpy()) ** 2)
        )
    )

    modele_temps = construire_gbm(parametres, cfg.graine).fit(
        decoupage.apprentissage[colonnes], decoupage.apprentissage[cible]
    )
    rmse_validation = float(
        np.sqrt(
            np.mean(
                (
                    modele_temps.predict(decoupage.validation[colonnes])
                    - decoupage.validation[cible].to_numpy()
                )
                ** 2
            )
        )
    )
    rmse_test = float(
        np.sqrt(
            np.mean(
                (modele_temps.predict(decoupage.test[colonnes]) - decoupage.test[cible].to_numpy())
                ** 2
            )
        )
    )

    resultat = {
        "rmse_decoupage_aleatoire": round(rmse_alea, 3),
        "rmse_saison_validation": round(rmse_validation, 3),
        "rmse_saison_test": round(rmse_test, 3),
        "surestimation_pct": round(100 * (rmse_test / rmse_alea - 1), 1),
        "lecture": (
            "Un decoupage aleatoire place des lignes de la meme heure et de stations voisines "
            "des deux cotes de la separation. Les variables atmospheriques y sont identiques : "
            "le modele retrouve en test des situations qu'il a deja vues en apprentissage."
        ),
    }
    _ecrire_json(cfg.chemin("artefacts_dir") / "experience_protocole.json", resultat)
    return resultat


def experience_pieges_methodologiques(cfg: Config, decoupage: Decoupage) -> dict:
    """Chiffre trois pièges classiques, pour montrer de combien ils trompent.

    Les trois ont été rencontrés sur la première version de ce travail, et
    aucun ne provoque d'erreur a l'exécution : le code tourne, les scores sont
    flatteurs, et la conclusion est fausse. Les chiffrer est la seule façon de
    s'en souvenir.

    1. **Cible erronée** : apprendre la température sèche du capteur
       (T2M_D_med) au lieu de la température humide. La tache devient bien plus
       facile, puisque la température sèche sans abri est presque la
       température sous abri décalée d'un degré et demi. Le score obtenu ne dit
       rien de la question posée.
    2. **Découpage aléatoire** : déja chiffré par
       `experience_protocole_evaluation`, rappelé ici pour comparaison.
    3. **Évaluation circulaire** : ajuster la formule de Stull sur les valeurs
       qu'elle produit elle-même, puis mesurer l'accord entre les deux. Le R2
       obtenu approche 1 par construction, sans qu'aucune mesure de terrain
       n'intervienne.
    """
    from sklearn.model_selection import train_test_split

    from .features import temperature_humide_stull
    from .models import construire_gbm

    colonnes = jeux_de_variables()["complet"]
    parametres = dict(cfg.modeles["gbm_grille"][0])
    complet = decoupage.apprentissage_complet

    # 1. Cible erronee, avec le decoupage aleatoire de la premiere version.
    train_alea, test_alea = train_test_split(
        complet, test_size=0.3, random_state=cfg.graine, shuffle=True
    )
    modele_faux = construire_gbm(parametres, cfg.graine).fit(
        train_alea[colonnes], train_alea["T2M_D_med"]
    )
    rmse_cible_erronee = float(
        np.sqrt(
            np.mean(
                (modele_faux.predict(test_alea[colonnes]) - test_alea["T2M_D_med"].to_numpy()) ** 2
            )
        )
    )

    # 3. Evaluation circulaire : on ajuste la formule sur ses propres sorties,
    # exactement comme le faisait la premiere version, puis on mesure l'accord.
    from .models import StullRecalibre

    test = decoupage.test
    sortie_formule_train = temperature_humide_stull(complet["T2M"], complet["RH2M"])
    modele_circulaire = StullRecalibre().fit(
        complet[["T2M", "RH2M"]], sortie_formule_train
    )
    sortie_formule_test = temperature_humide_stull(test["T2M"], test["RH2M"])
    prediction_circulaire = modele_circulaire.predict(test[["T2M", "RH2M"]])
    mesure = test[cfg.cible].to_numpy(dtype="float64")

    r2_circulaire = 1 - np.mean(
        (prediction_circulaire - sortie_formule_test) ** 2
    ) / np.var(sortie_formule_test)
    rmse_circulaire = float(np.sqrt(np.mean((prediction_circulaire - sortie_formule_test) ** 2)))
    r2_reel = 1 - np.mean((sortie_formule_test - mesure) ** 2) / np.var(mesure)

    resultat = {
        "cible_erronee": {
            "cible_apprise": "T2M_D_med (temperature seche du capteur)",
            "rmse": round(rmse_cible_erronee, 3),
            "lecture": (
                "La temperature seche sans abri se deduit presque directement de la "
                "temperature sous abri : le probleme resolu n'est pas celui pose."
            ),
        },
        "evaluation_circulaire": {
            "r2_formule_contre_elle_meme": round(float(r2_circulaire), 6),
            "rmse_formule_contre_elle_meme": round(rmse_circulaire, 4),
            "r2_formule_contre_la_mesure": round(float(r2_reel), 3),
            "lecture": (
                "Ajuster la formule sur ses propres sorties donne un R2 de 1 par "
                "construction. Confrontee aux mesures de terrain, la meme formule "
                "explique nettement moins de variance."
            ),
        },
    }
    _ecrire_json(cfg.chemin("artefacts_dir") / "experience_pieges.json", resultat)
    return resultat


def experience_stations_inconnues(
    cfg: Config, decoupage: Decoupage, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compare la performance sur capteurs déja vus et sur capteurs nouveaux.

    La période de test contient des stations absentes de l'apprentissage. C'est
    une expérience naturelle qui répond a la question produit : peut-on servir
    une parcelle qui vient d'être équipée, ou pire, une parcelle sans capteur ?
    """
    cible = cfg.cible
    connues = set(decoupage.apprentissage_complet["station_id"].unique())
    est_connue = decoupage.test["station_id"].isin(connues).to_numpy()
    y_test = decoupage.test[cible].to_numpy(dtype="float64")

    lignes = []
    for nom, prediction in predictions.items():
        for libelle, masque in (("stations connues", est_connue), ("stations nouvelles", ~est_connue)):
            if masque.sum() < 100:
                continue
            erreur = prediction[masque] - y_test[masque]
            lignes.append(
                {
                    "modele": nom,
                    "population": libelle,
                    "lignes": int(masque.sum()),
                    "stations": int(decoupage.test.loc[masque, "station_id"].nunique()),
                    "rmse": float(np.sqrt(np.mean(erreur**2))),
                    "mae": float(np.mean(np.abs(erreur))),
                    "biais": float(np.mean(erreur)),
                }
            )
    table = pd.DataFrame(lignes)
    table.to_csv(cfg.chemin("artefacts_dir") / "stations_connues_vs_nouvelles.csv", index=False)
    return table


def importances_permutation(
    cfg: Config, decoupage: Decoupage, modele: ModeleEvalue, lignes_max: int = 80_000
) -> pd.DataFrame:
    """Importance par permutation sur un échantillon de la période de test.

    L'importance par permutation mesure la dégradation réelle de l'erreur quand
    on casse le lien entre une variable et la cible. Contrairement aux
    importances internes d'un arbre, elle est calculée sur des données non vues
    et reste comparable d'un modèle a l'autre.
    """
    from sklearn.inspection import permutation_importance

    echantillon = decoupage.test.sample(
        n=min(lignes_max, len(decoupage.test)), random_state=cfg.graine
    )
    resultat = permutation_importance(
        modele.estimateur,
        echantillon[modele.colonnes],
        echantillon[cfg.cible],
        n_repeats=5,
        random_state=cfg.graine,
        scoring="neg_root_mean_squared_error",
    )
    table = pd.DataFrame(
        {
            "variable": modele.colonnes,
            "importance": resultat.importances_mean,
            "ecart_type": resultat.importances_std,
        }
    ).sort_values("importance", ascending=False)
    table.to_csv(cfg.chemin("artefacts_dir") / "importances_gbm.csv", index=False)
    viz.figure_importances(
        table,
        cfg.chemin("figures_dir") / "importances_gbm.png",
        "Variables qui portent la prédiction (gradient boosting)",
    )
    return table


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def executer_tout(cfg: Config, refaire_nettoyage: bool = False) -> dict:
    """Enchaine toutes les étapes et écrit le résumé global."""
    depart = time.perf_counter()
    if refaire_nettoyage or not cfg.chemin("clean_parquet").exists():
        etape_nettoyage(cfg)

    decoupage = preparer_decoupage(cfg)
    resume_decoupage = decoupage.resume()
    resume_decoupage.to_csv(cfg.chemin("artefacts_dir") / "resume_decoupage.csv", index=False)
    LOGGER.info("decoupage :\n%s", resume_decoupage.to_string(index=False))

    modeles, traces = entrainer_modeles(cfg, decoupage)
    resultats = evaluer(cfg, decoupage, modeles)
    LOGGER.info(
        "metriques de test :\n%s",
        resultats["tableau"][
            ["modele", "rmse", "mae", "biais", "depassement_1C_pct", "pod_rappel_gel", "far_gel"]
        ].to_string(index=False),
    )

    protocole = experience_protocole_evaluation(cfg, decoupage)
    pieges = experience_pieges_methodologiques(cfg, decoupage)
    stations = experience_stations_inconnues(cfg, decoupage, resultats["predictions"])
    gbm = next(m for m in modeles if m.nom == "Gradient boosting")
    importances = importances_permutation(cfg, decoupage, gbm)

    resume = {
        "duree_totale_s": round(time.perf_counter() - depart, 1),
        "traces_apprentissage": traces,
        "experience_protocole": protocole,
        "experience_pieges": pieges,
        "modeles": [
            {"nom": m.nom, "description": m.description, "colonnes": m.colonnes, "details": _serialisable(m.details)}
            for m in modeles
        ],
    }
    _ecrire_json(cfg.chemin("artefacts_dir") / "resume_execution.json", resume)

    return {
        "decoupage": resume_decoupage,
        "modeles": modeles,
        "resultats": resultats,
        "protocole": protocole,
        "pieges": pieges,
        "stations": stations,
        "importances": importances,
        "resume": resume,
    }


def _serialisable(valeur):
    """Convertit les types numpy en types Python pour l'écriture JSON."""
    if isinstance(valeur, dict):
        return {str(cle): _serialisable(val) for cle, val in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [_serialisable(val) for val in valeur]
    if isinstance(valeur, (np.integer,)):
        return int(valeur)
    if isinstance(valeur, (np.floating,)):
        return float(valeur)
    return valeur


def _ecrire_json(chemin: Path, contenu: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(_serialisable(contenu), indent=2, ensure_ascii=False), encoding="utf-8"
    )
