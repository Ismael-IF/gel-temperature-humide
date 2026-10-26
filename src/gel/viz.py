"""Figures du rapport.

Trois règles appliquées partout, parce qu'elles décident de la lisibilité bien
plus que le choix des couleurs :

- une seule échelle par graphique, jamais deux axes verticaux ;
- comparer quatre modèles se fait en petits multiples, pas en superposant
  quatre courbes de distribution ;
- la couleur identifie un modèle et ne change jamais d'un graphique a l'autre,
  pour que le lecteur n'ait pas a relire la légende a chaque figure.

La palette est fixée en tête de module : teintes catégorielles séparables, y
compris pour un lecteur daltonien, et une rampe bleue pour les magnitudes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
TEXTE = "#0b0b0b"
TEXTE_SECONDAIRE = "#52514e"
GRILLE = "#e3e2de"

# Teintes catégorielles, assignées par modèle une fois pour toutes.
COULEURS_MODELES: dict[str, str] = {
    "Persistance 2 m": "#e34948",
    "Stull (coefficients publiés)": "#eda100",
    "Stull recalibré": "#1baf7a",
    "Régression polynomiale": "#eb6834",
    "Gradient boosting": "#2a78d6",
}
BLEU = "#2a78d6"
BLEU_CLAIR = "#9ec5f4"
RAMPE_BLEUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def appliquer_style() -> None:
    """Style commun : fond clair, grille discrète, encre sobre."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRILLE,
            "axes.labelcolor": TEXTE_SECONDAIRE,
            "axes.titlecolor": TEXTE,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRILLE,
            "grid.linewidth": 0.8,
            "xtick.color": TEXTE_SECONDAIRE,
            "ytick.color": TEXTE_SECONDAIRE,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def couleur(modele: str, defaut: str = BLEU) -> str:
    return COULEURS_MODELES.get(modele, defaut)


def _enregistrer(fig: plt.Figure, chemin: Path) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


def _sans_cadre(ax: plt.Axes, garder: tuple[str, ...] = ("left", "bottom")) -> None:
    for cote, epine in ax.spines.items():
        epine.set_visible(cote in garder)


def figure_journal_nettoyage(journal: pd.DataFrame, chemin: Path) -> Path:
    """Volume restant après chaque filtre : ou part réellement la donnée."""
    appliquer_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    etiquettes = journal["filtre"].tolist()
    restantes = journal["lignes_restantes"].to_numpy() / 1e6
    initiales = journal.attrs.get("lignes_initiales", journal["lignes_restantes"].iloc[0]) / 1e6

    positions = np.arange(len(etiquettes) + 1)
    valeurs = np.concatenate([[initiales], restantes])
    libelles = ["données brutes"] + etiquettes

    barres = ax.barh(positions, valeurs, color=[BLEU_CLAIR] + [BLEU] * len(etiquettes), height=0.62)
    barres[0].set_color("#c3c2b7")
    for position, valeur in zip(positions, valeurs):
        ax.text(
            valeur + initiales * 0.012,
            position,
            f"{valeur:.2f} M",
            va="center",
            fontsize=8.5,
            color=TEXTE_SECONDAIRE,
        )

    ax.set_yticks(positions, libelles)
    ax.invert_yaxis()
    ax.set_xlabel("lignes restantes (millions)")
    ax.set_title("Réduction du volume, filtre par filtre")
    ax.set_xlim(0, initiales * 1.12)
    ax.grid(axis="y", visible=False)
    _sans_cadre(ax)
    return _enregistrer(fig, chemin)


def figure_distribution_erreurs(
    erreurs: dict[str, np.ndarray], chemin: Path, borne: float = 4.0
) -> Path:
    """Petits multiples : une distribution d'erreur par modèle, même échelle.

    Superposer quatre densités rendrait les queues illisibles, et ce sont
    justement les queues qui portent le risque agronomique.
    """
    appliquer_style()
    noms = list(erreurs)
    fig, axes = plt.subplots(1, len(noms), figsize=(3.1 * len(noms), 3.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    bornes = np.linspace(-borne, borne, 81)

    for ax, nom in zip(axes, noms):
        serie = np.clip(erreurs[nom], -borne, borne)
        ax.hist(serie, bins=bornes, color=couleur(nom), alpha=0.9)
        ax.axvline(0, color=TEXTE_SECONDAIRE, linewidth=1.0, linestyle=(0, (4, 3)))
        rmse = float(np.sqrt(np.mean(erreurs[nom] ** 2)))
        biais = float(np.mean(erreurs[nom]))
        ax.set_title(nom, fontsize=9.5)
        ax.text(
            0.03,
            0.95,
            f"RMSE {rmse:.2f} °C\nbiais {biais:+.2f} °C",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color=TEXTE_SECONDAIRE,
        )
        ax.set_xlabel("erreur de prédiction (°C)")
        ax.grid(axis="x", visible=False)
        _sans_cadre(ax)

    axes[0].set_ylabel("nombre d'heures")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f} k"))
    fig.suptitle(
        "Distribution des erreurs sur la période de test (erreur = prédiction moins mesure)\n"
        f"les erreurs au-dela de ±{borne:g} °C sont ramenées sur les barres extrêmes",
        fontsize=10.5,
        color=TEXTE,
        y=1.08,
    )
    return _enregistrer(fig, chemin)


def figure_predictions_vs_mesures(
    predictions: dict[str, np.ndarray], mesure: np.ndarray, chemin: Path
) -> Path:
    """Nuage de densité prediction / mesure, avec la diagonale et le seuil de gel."""
    appliquer_style()
    noms = list(predictions)
    fig, axes = plt.subplots(1, len(noms), figsize=(3.3 * len(noms), 3.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    limites = (-10, 12)

    for ax, nom in zip(axes, noms):
        ax.hexbin(
            mesure,
            predictions[nom],
            gridsize=55,
            bins="log",
            cmap="Blues",
            extent=(*limites, *limites),
            mincnt=1,
        )
        ax.plot(limites, limites, color=TEXTE, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.axhline(0, color="#e34948", linewidth=0.9)
        ax.axvline(0, color="#e34948", linewidth=0.9)
        ax.set_title(nom, fontsize=9.5)
        ax.set_xlabel("température humide mesurée (°C)")
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.set_aspect("equal")
        _sans_cadre(ax)

    axes[0].set_ylabel("température humide prédite (°C)")
    fig.suptitle(
        "Prédictions contre mesures. En haut à gauche : il gelait, le modèle n'alertait pas",
        fontsize=10.5,
        color=TEXTE,
        y=1.02,
    )
    return _enregistrer(fig, chemin)


def figure_metriques_par_groupe(
    tables: dict[str, pd.DataFrame],
    colonne_groupe: str,
    chemin: Path,
    titre: str,
    etiquette_x: str,
    metrique: str = "rmse",
) -> Path:
    """RMSE par mois ou par heure : une ligne par modèle, une seule échelle."""
    appliquer_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.0))

    for nom, table in tables.items():
        ax.plot(
            table[colonne_groupe],
            table[metrique],
            marker="o",
            color=couleur(nom),
            label=nom,
        )
    # Pas d'etiquette en bout de courbe ici : les quatre modeles convergent sur
    # les dernieres heures, les etiquettes se superposeraient. La legende suffit.
    ax.set_xlabel(etiquette_x)
    ax.set_ylabel(f"{metrique.upper()} (°C)")
    ax.set_title(titre)
    ax.legend(loc="upper left", ncols=2)
    ax.margins(x=0.14)
    _sans_cadre(ax)
    return _enregistrer(fig, chemin)


def figure_contingences_gel(
    contingences: dict[str, dict[str, float]], chemin: Path
) -> Path:
    """Tables de contingence du classement gel, en effectifs et en part de ligne."""
    appliquer_style()
    noms = list(contingences)
    fig, axes = plt.subplots(1, len(noms), figsize=(3.0 * len(noms), 3.2))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.25)

    for ax, nom in zip(axes, noms):
        c = contingences[nom]
        matrice = np.array(
            [[c["vrais_positifs"], c["faux_negatifs"]], [c["faux_positifs"], c["vrais_negatifs"]]],
            dtype="float64",
        )
        parts = matrice / matrice.sum(axis=1, keepdims=True)
        ax.imshow(parts, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{int(matrice[i, j]):,}".replace(",", " ") + f"\n{100*parts[i, j]:.1f} %",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=TEXTE if parts[i, j] < 0.55 else SURFACE,
                )
        ax.set_xticks([0, 1], ["alerte", "pas d'alerte"], fontsize=8.5)
        # Les libelles de lignes ne sont portes que par le premier panneau :
        # repetes, ils se superposent aux cellules du panneau precedent.
        premier = ax is axes[0]
        ax.set_yticks([0, 1], ["gel observé", "pas de gel"] if premier else ["", ""], fontsize=8.5)
        ax.set_title(f"{nom}\nPOD {100*c['pod_rappel']:.0f} %  FAR {100*c['far_fausses_alertes']:.0f} %", fontsize=9)
        ax.grid(visible=False)
        _sans_cadre(ax, garder=())

    fig.suptitle(
        "Détection des nuits de gel (minimum nocturne sous 0 °C), par station et par nuit",
        fontsize=10.5,
        color=TEXTE,
        y=1.04,
    )
    return _enregistrer(fig, chemin)


def figure_cout_par_seuil(
    balayages: dict[str, pd.DataFrame], chemin: Path, surface_ha: float = 1.0
) -> Path:
    """Cout des erreurs en fonction du seuil d'alerte, et seuil optimal par modèle."""
    appliquer_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    # Les quatre optimums sont tres proches : on les marque tous, mais une seule
    # annotation, sinon les etiquettes se chevauchent et deviennent illisibles.
    optimums = {
        nom: table.loc[table["cout_total_eur"].idxmin()] for nom, table in balayages.items()
    }
    meilleur_global = min(optimums, key=lambda nom: optimums[nom]["cout_total_eur"])

    for nom, table in balayages.items():
        ax.plot(table["seuil_alerte"], table["cout_total_eur"] / 1000, color=couleur(nom), label=nom)
        meilleur = optimums[nom]
        ax.plot(
            meilleur["seuil_alerte"],
            meilleur["cout_total_eur"] / 1000,
            marker="o",
            markersize=8,
            color=couleur(nom),
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )

    optimum = optimums[meilleur_global]
    seuil_optimal = f"{optimum['seuil_alerte']:+.1f}".replace(".", ",")
    ax.annotate(
        f"cout minimal à {seuil_optimal} °C\n"
        f"({100 * optimum['pod_rappel']:.0f} % des nuits de gel détectées)",
        (optimum["seuil_alerte"], optimum["cout_total_eur"] / 1000),
        textcoords="offset points",
        xytext=(-10, 48),
        ha="right",
        fontsize=8.5,
        color=TEXTE_SECONDAIRE,
        arrowprops={"arrowstyle": "-", "color": TEXTE_SECONDAIRE, "linewidth": 0.8},
    )

    ax.set_xlabel("seuil de déclenchement de l'alerte sur le minimum nocturne prédit (°C)")
    ax.set_ylabel("coût des erreurs (milliers d'euros)")
    ax.set_title(
        "Coût des erreurs de décision selon le seuil d'alerte\n"
        f"maille station-nuit, {surface_ha:g} ha par station, printemps 2023"
    )
    ax.legend(loc="upper right", ncols=1)
    _sans_cadre(ax)
    return _enregistrer(fig, chemin)


def figure_erreur_par_regime(
    tables: dict[str, pd.DataFrame], chemin: Path, colonne_groupe: str = "regime"
) -> Path:
    """RMSE par régime de température : la performance utile est celle du froid."""
    appliquer_style()
    noms = list(tables)
    categories = tables[noms[0]][colonne_groupe].astype(str).tolist()
    positions = np.arange(len(categories))
    largeur = 0.8 / len(noms)

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for index, nom in enumerate(noms):
        decalage = (index - (len(noms) - 1) / 2) * largeur
        ax.bar(
            positions + decalage,
            tables[nom]["rmse"],
            width=largeur * 0.92,
            color=couleur(nom),
            label=nom,
        )
    ax.set_xticks(positions, categories)
    ax.set_xlabel("température humide mesurée (°C)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Erreur par régime de température, période de test")
    ax.legend(ncols=2)
    ax.grid(axis="x", visible=False)
    _sans_cadre(ax)
    return _enregistrer(fig, chemin)


def figure_importances(importances: pd.DataFrame, chemin: Path, titre: str) -> Path:
    """Importance par permutation : contribution mesurée, pas structure de l'arbre."""
    appliquer_style()
    table = importances.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(table) + 1.6))
    ax.barh(table["variable"], table["importance"], color=BLEU, height=0.62)
    for nom, valeur in zip(table["variable"], table["importance"]):
        ax.text(
            valeur + table["importance"].max() * 0.015,
            nom,
            f"{valeur:.3f}",
            va="center",
            fontsize=8,
            color=TEXTE_SECONDAIRE,
        )
    ax.set_xlabel("augmentation de la RMSE quand la variable est permutée (°C)")
    ax.set_title(titre)
    ax.set_xlim(0, table["importance"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    _sans_cadre(ax)
    return _enregistrer(fig, chemin)
