"""Chargement de la configuration du projet.

Un seul fichier TOML porte les chemins, les seuils de nettoyage, le découpage
temporel et les hypothèses économiques. Le code ne contient aucune valeur
magique : tout ce qui pourrait être rediscuté avec le métier est dans le TOML.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parents[2]
CONFIG_PAR_DEFAUT = RACINE / "config" / "config.toml"


@dataclass(frozen=True)
class Config:
    """Vue typée de config.toml, avec résolution des chemins relatifs."""

    brut: dict[str, Any] = field(repr=False)
    racine: Path = RACINE

    # --- chemins ---------------------------------------------------------
    def chemin(self, cle: str) -> Path:
        """Renvoie un chemin de la section [paths], relatif à la racine du projet."""
        valeur = Path(self.brut["paths"][cle])
        return valeur if valeur.is_absolute() else self.racine / valeur

    # --- raccourcis de lecture ------------------------------------------
    @property
    def features(self) -> list[str]:
        """Les 10 variables atmosphériques autorisées par l'énoncé."""
        return list(self.brut["colonnes"]["atmospheriques"])

    @property
    def cible(self) -> str:
        return self.brut["colonnes"]["cible"]

    @property
    def nettoyage(self) -> dict[str, Any]:
        return self.brut["nettoyage"]

    @property
    def modeles(self) -> dict[str, Any]:
        return self.brut["modeles"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.brut["evaluation"]

    @property
    def metier(self) -> dict[str, Any]:
        return self.brut["metier"]

    @property
    def graine(self) -> int:
        return int(self.brut["modeles"]["graine"])

    def date(self, cle: str) -> dt.date:
        """Renvoie une borne de la section [decoupage] (TOML type date natif)."""
        return self.brut["decoupage"][cle]


def charger_config(chemin: str | Path | None = None) -> Config:
    """Lit le fichier TOML de configuration et renvoie un objet Config."""
    chemin = Path(chemin) if chemin else CONFIG_PAR_DEFAUT
    with chemin.open("rb") as flux:
        return Config(brut=tomllib.load(flux))
