"""Entrées / sorties : conversion du CSV brut en Parquet et lectures typées.

Le fichier brut pèse 1,4 Go pour 11,8 M de lignes. Le lire en float64 à chaque
exécution coûte plusieurs minutes et sature la mémoire. On le convertit donc
une fois pour toutes en Parquet avec des types compacts (float32 / int32), ce
qui ramène le jeu de données à une trentaine de Mo et la lecture à quelques
secondes. C'est aussi le format demandé en livrable du projet 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Colonnes du jeu de données et types retenus. Les mesures météo ont au plus
# une décimale : float32 (7 chiffres significatifs) est largement suffisant et
# divise par deux l'empreinte mémoire.
TYPES_MESURES = {
    "hour": "int8",
    "station_id": "int32",
    "nbre_records": "int16",
    "T2M": "float32",
    "DEWT2M": "float32",
    "RH2M": "float32",
    "WS2M": "float32",
    "WS10M": "float32",
    "WD": "float32",
    "VPD": "float32",
    "CLDC": "float32",
    "SSI": "float32",
    "PRECIP": "float32",
    "lat": "float64",  # la géolocalisation sert à un filtre au 0,05 degré : on garde la précision
    "lon": "float64",
    "T2M_D_med": "float32",
    "T2M_U_med": "float32",
}


def convertir_csv_en_parquet(csv: Path, parquet: Path, taille_bloc: int = 2_000_000) -> int:
    """Convertit le CSV brut en Parquet typé, par blocs pour borner la mémoire.

    Renvoie le nombre de lignes écrites. La colonne d'index anonyme du CSV
    ("Unnamed: 0") est supprimée : c'est un artefact d'export, pas une donnée.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet.parent.mkdir(parents=True, exist_ok=True)
    lecteur = pd.read_csv(
        csv,
        dtype=TYPES_MESURES,
        parse_dates=["date"],
        index_col=0,
        chunksize=taille_bloc,
    )

    ecrivain: pq.ParquetWriter | None = None
    lignes = 0
    try:
        for bloc in lecteur:
            table = pa.Table.from_pandas(bloc.reset_index(drop=True), preserve_index=False)
            if ecrivain is None:
                ecrivain = pq.ParquetWriter(parquet, table.schema, compression="zstd")
            ecrivain.write_table(table)
            lignes += len(bloc)
            LOGGER.info("%s lignes converties", f"{lignes:,}".replace(",", " "))
    finally:
        if ecrivain is not None:
            ecrivain.close()
    return lignes


def lire_parquet(chemin: Path, colonnes: list[str] | None = None) -> pd.DataFrame:
    """Lecture Parquet, avec projection de colonnes quand elle est possible."""
    return pd.read_parquet(chemin, columns=colonnes)


def ecrire_parquet(df: pd.DataFrame, chemin: Path) -> None:
    """Écrit un DataFrame en Parquet compressé, sans index technique."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chemin, index=False, compression="zstd")
