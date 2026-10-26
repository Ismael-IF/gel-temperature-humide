"""Interface en ligne de commande : `python -m gel <etape>`.

Les étapes sont indépendantes et s'enchainent par les fichiers qu'elles écrivent,
ce qui permet de relancer la modélisation sans refaire le nettoyage (quelques
secondes contre une dizaine de minutes).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import charger_config
from .io import convertir_csv_en_parquet
from .pipeline import (
    etape_nettoyage,
    entrainer_modeles,
    evaluer,
    executer_tout,
    experience_pieges_methodologiques,
    experience_protocole_evaluation,
    preparer_decoupage,
)


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        prog="python -m gel",
        description="Prevision de la temperature humide pour l'alerte gel.",
    )
    parseur.add_argument(
        "etape",
        choices=[
            "ingestion",
            "nettoyage",
            "modelisation",
            "protocole",
            "pieges",
            "tout",
        ],
        help=(
            "ingestion : CSV brut vers Parquet typé | "
            "nettoyage : filtres du projet 1 | "
            "modelisation : apprentissage et evaluation des 3 approches | "
            "protocole : comparaison decoupage aleatoire / chronologique | "
            "pieges : chiffrage des pieges methodologiques | "
            "tout : la chaine complete"
        ),
    )
    parseur.add_argument("--config", type=Path, default=None, help="chemin d'un autre config.toml")
    parseur.add_argument(
        "--refaire-nettoyage",
        action="store_true",
        help="force le nettoyage meme si le Parquet nettoye existe deja",
    )
    return parseur


def principal(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # matplotlib journalise en INFO des details de rendu sans interet ici.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = charger_config(arguments.config)

    if arguments.etape == "ingestion":
        lignes = convertir_csv_en_parquet(cfg.chemin("raw_csv"), cfg.chemin("raw_parquet"))
        logging.info("%d lignes converties vers %s", lignes, cfg.chemin("raw_parquet"))
    elif arguments.etape == "nettoyage":
        etape_nettoyage(cfg)
    elif arguments.etape == "modelisation":
        decoupage = preparer_decoupage(cfg)
        modeles, _ = entrainer_modeles(cfg, decoupage)
        resultats = evaluer(cfg, decoupage, modeles)
        print(resultats["tableau"].to_string(index=False))
    elif arguments.etape == "protocole":
        print(experience_protocole_evaluation(cfg, preparer_decoupage(cfg)))
    elif arguments.etape == "pieges":
        print(experience_pieges_methodologiques(cfg, preparer_decoupage(cfg)))
    else:
        executer_tout(cfg, refaire_nettoyage=arguments.refaire_nettoyage)
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
