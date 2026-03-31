"""
ASR Español (España) — script de entrenamiento de reconocimiento automático
del habla orientado a variedades peninsulares.

Datasets objetivo:
  • Common Voice 16.0 ES  — mozilla-foundation/common_voice_16_0 (config "es")
  • RTVE 2022 ASR          — projecte-aina/rtve2022asr
"""

from __future__ import annotations

import logging
import pathlib
import random
import sys
import tarfile
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acentos / variantes geográficas marcadas en Common Voice que NO corresponden
# a variedades peninsulares del español.
# ---------------------------------------------------------------------------
_NON_PENINSULAR_ACCENTS = {
    "Mexican Spanish",
    "Rioplatense Spanish",
    "Colombian Spanish",
    "Venezuelan Spanish",
    "Chilean Spanish",
    "Peruvian Spanish",
    "Cuban Spanish",
    "Puerto Rican Spanish",
    "Dominican Spanish",
    "Bolivian Spanish",
    "Paraguayan Spanish",
    "Uruguayan Spanish",
    "Ecuadorian Spanish",
    "Guatemalan Spanish",
    "Honduran Spanish",
    "Nicaraguan Spanish",
    "Costa Rican Spanish",
    "Panamanian Spanish",
    "Salvadoran Spanish",
}


def _is_peninsular(sample: Dict) -> bool:
    """Devuelve True si la muestra pertenece a un hablante peninsular.

    Se aplica a cada fila del dataset de Common Voice.  Si el campo ``accent``
    está vacío o no existe se acepta la muestra (beneficio de la duda).
    """
    accent = (sample.get("accent") or "").strip()
    if not accent:
        return True
    return accent not in _NON_PENINSULAR_ACCENTS


def _hf_to_samples(dataset: Any, split: str, source: str) -> List[Dict]:
    """Convierte un split de HuggingFace Dataset a la lista interna de muestras.

    Cada elemento de la lista tiene al menos:
      • ``sentence``  — transcripción en texto
      • ``audio``     — dict con ``array`` y ``sampling_rate``  (o ``path``)
      • ``source``    — origen del dato ("common_voice" | "rtve2022")
    """
    samples = []
    for row in dataset[split]:
        samples.append({**row, "source": source})
    return samples


# ---------------------------------------------------------------------------
# Descarga de Common Voice ES
# ---------------------------------------------------------------------------

def download_common_voice_spain(data_dir: str = "data/common_voice") -> Optional[Any]:
    """Descarga Common Voice 16.0 ES y filtra solo hablantes peninsulares.

    Excluye muestras de México, Argentina, Colombia, etc.

    Returns:
        DatasetDict de HuggingFace o ``None`` si la descarga falló.
    """
    data_dir_path = pathlib.Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    log.info("Descargando Common Voice ES (Common Voice 16.0)...")
    try:
        from datasets import load_dataset  # noqa: PLC0415

        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_0",
            "es",
            cache_dir=str(data_dir_path / "hf_cache"),
        )

        log.info("Filtrando hablantes peninsulares en Common Voice...")
        for split in list(dataset.keys()):
            original = len(dataset[split])
            dataset[split] = dataset[split].filter(_is_peninsular)
            log.info(
                "  %s: %d → %d muestras peninsulares",
                split,
                original,
                len(dataset[split]),
            )

        sizes = {s: len(dataset[s]) for s in dataset.keys()}
        log.info("✓ Common Voice: %s", sizes)
        return dataset

    except Exception as exc:
        log.error("Common Voice no disponible: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Descarga de RTVE 2022
# ---------------------------------------------------------------------------

def download_rtve2022(data_dir: str = "data/rtve2022") -> Optional[str]:
    """Descarga el corpus RTVE 2022 (grabaciones TV/radio española).

    Fuentes (en orden de preferencia):
      1. HuggingFace: projecte-aina/rtve2022asr
      2. Zenodo IberSPEECH 2022

    Returns:
        Ruta al directorio con los datos, o ``None`` si falló todo.
    """
    data_dir_path = pathlib.Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    done_file = data_dir_path / ".downloaded"

    if done_file.exists():
        log.info("RTVE 2022 ya disponible en %s", data_dir_path)
        return str(data_dir_path)

    # -- Intento 1: HuggingFace ------------------------------------------------
    log.info("Intentando descargar RTVE 2022 desde HuggingFace...")
    try:
        from datasets import load_dataset  # noqa: PLC0415

        dataset = load_dataset(
            "projecte-aina/rtve2022asr",
            cache_dir=str(data_dir_path / "hf_cache"),
        )
        done_file.write_text("huggingface")
        log.info("RTVE 2022 cargado desde HuggingFace: %s", list(dataset.keys()))
        return str(data_dir_path)

    except Exception as exc:
        log.error("HuggingFace RTVE falló: %s", exc)

    # -- Intento 2: Zenodo IberSPEECH 2022 ------------------------------------
    log.info("Intentando descarga desde Zenodo (IberSPEECH 2022)...")
    try:
        import requests  # noqa: PLC0415

        zenodo_url = "https://zenodo.org/record/7248553/files/rtve2022asr.tar.gz"
        # Timeout para la conexión inicial; la descarga completa (~5 GB) puede
        # tardar varios minutos dependiendo de la velocidad de red.
        r = requests.get(zenodo_url, stream=True, timeout=300)
        if r.status_code == 200:
            tar_path = data_dir_path / "rtve2022asr.tar.gz"
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65_536):
                    f.write(chunk)
            with tarfile.open(tar_path) as tf:
                tf.extractall(data_dir_path)
            done_file.write_text("zenodo")
            log.info("RTVE 2022 descargado en %s", data_dir_path)
            return str(data_dir_path)

        log.error("Zenodo devolvió HTTP %d", r.status_code)

    except Exception as exc:
        log.error("Descarga Zenodo fallida: %s", exc)

    log.warning(
        "No fue posible descargar RTVE 2022 automáticamente.\n"
        "Por favor, descárgalo manualmente desde:\n"
        "  https://zenodo.org/record/7248553\n"
        "y colócalo en: %s",
        data_dir_path,
    )
    return None


# ---------------------------------------------------------------------------
# Carga de RTVE 2022 desde caché HuggingFace
# ---------------------------------------------------------------------------

def load_rtve2022_hf(data_dir: str = "data/rtve2022") -> Optional[Any]:
    """Carga el dataset RTVE 2022 si ya está en caché de HuggingFace.

    Returns:
        DatasetDict de HuggingFace o ``None`` si no está disponible.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415

        return load_dataset(
            "projecte-aina/rtve2022asr",
            cache_dir=str(pathlib.Path(data_dir) / "hf_cache"),
        )

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Construcción de DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 8,
    num_workers: int = 0,
    val_fraction: float = 0.05,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Descarga (si necesario) y prepara las muestras de entrenamiento y
    validación usando Common Voice ES + RTVE 2022.

    Returns:
        Tupla ``(train_samples, val_samples, corpus_texts)``.

    Raises:
        SystemExit: Si no se encuentran datos reales.  No usamos datos
            sintéticos porque no son aptos para entrenar un ASR en producción.
    """
    all_samples: List[Dict] = []
    corpus_texts: List[str] = []
    cv_downloaded = False
    rtve_downloaded = False

    log.info("\n[FASE 6] Descargando y preparando datasets...")

    # -- Common Voice ES -------------------------------------------------------
    cv = download_common_voice_spain(f"{data_dir}/common_voice")
    if cv is not None:
        cv_split_sizes: Dict[str, int] = {}
        for split in ("train", "validation", "test"):
            if split in cv:
                ss = _hf_to_samples(cv, split, "common_voice")
                all_samples.extend(ss)
                corpus_texts.extend(s["sentence"] for s in ss)
                cv_split_sizes[split] = len(ss)
        cv_downloaded = True
        log.info(
            "✓ Common Voice: train=%d, validation=%d, test=%d",
            cv_split_sizes.get("train", 0),
            cv_split_sizes.get("validation", 0),
            cv_split_sizes.get("test", 0),
        )
    else:
        log.warning("✗ Common Voice ES: Descarga fallida")

    # -- RTVE 2022 ------------------------------------------------------------
    rtve_dir = download_rtve2022(f"{data_dir}/rtve2022")
    if rtve_dir:
        rtve = load_rtve2022_hf(rtve_dir)
        if rtve is not None:
            rtve_split_sizes: Dict[str, int] = {}
            for split in ("train", "validation", "test"):
                if split in rtve:
                    ss = _hf_to_samples(rtve, split, "rtve2022")
                    all_samples.extend(ss)
                    corpus_texts.extend(s["sentence"] for s in ss)
                    rtve_split_sizes[split] = len(ss)
            rtve_downloaded = True
            log.info(
                "✓ RTVE 2022: train=%d, validation=%d, test=%d",
                rtve_split_sizes.get("train", 0),
                rtve_split_sizes.get("validation", 0),
                rtve_split_sizes.get("test", 0),
            )
        else:
            log.warning("✗ RTVE 2022: No se pudo cargar desde caché HuggingFace")
    else:
        log.warning("✗ RTVE 2022: Descarga fallida")

    # -- Verificación: se requieren datos reales ------------------------------
    if not all_samples:
        _abort_no_real_data(data_dir, cv_downloaded, rtve_downloaded)

    log.info("Total muestras: %d", len(all_samples))

    # -- Split train / val ----------------------------------------------------
    random.shuffle(all_samples)
    n_val = max(50, int(len(all_samples) * val_fraction))
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]
    log.info("Split → Train: %d  |  Val: %d", len(train_samples), len(val_samples))

    return train_samples, val_samples, corpus_texts


def _abort_no_real_data(
    data_dir: str,
    cv_downloaded: bool,
    rtve_downloaded: bool,
) -> None:
    """Imprime instrucciones de descarga manual y termina el proceso."""
    log.error("=" * 70)
    log.error("ERROR CRÍTICO: NO SE ENCONTRARON DATOS REALES")
    log.error("=" * 70)
    log.error(
        "Este script requiere datos reales. "
        "Los datos sintéticos NO son aceptables para ASR en producción."
    )

    if not cv_downloaded:
        log.error(
            "\n📥 Common Voice 16.0 ES NO está disponible (~15 GB):\n"
            "   1. Visita: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0\n"
            "   2. Acepta los términos de uso\n"
            "   3. Descarga con:\n"
            "      huggingface-cli download mozilla-foundation/common_voice_16_0 \\\n"
            "          --repo-type dataset\n"
            "   4. O coloca el caché en: %s/common_voice/hf_cache/",
            data_dir,
        )

    if not rtve_downloaded:
        log.error(
            "\n📥 RTVE 2022 ASR NO está disponible (~5 GB):\n"
            "   Opción A — HuggingFace:\n"
            "      huggingface-cli download projecte-aina/rtve2022asr --repo-type dataset\n"
            "      URL: https://huggingface.co/datasets/projecte-aina/rtve2022asr\n"
            "   Opción B — Zenodo (IberSPEECH 2022):\n"
            "      https://zenodo.org/record/7248553  →  rtve2022asr.tar.gz\n"
            "   Coloca los datos en: %s/rtve2022/",
            data_dir,
        )

    log.error(
        "\n📋 Pasos generales:\n"
        "   1. Inicia sesión en HuggingFace CLI:  huggingface-cli login\n"
        "   2. Descarga los datasets con los comandos anteriores\n"
        "   3. Vuelve a ejecutar este script"
    )
    sys.exit(1)
