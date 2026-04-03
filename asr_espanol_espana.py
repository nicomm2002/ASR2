#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASSR Español de España
======================
Sistema completo de reconocimiento automático de voz (ASR/ASSR)
para el español peninsular (castellano de España).

Arquitectura basada en:
  · Whisper        (OpenAI, 2022)       arxiv:2212.04356
  · Conformer      (Google, 2020)       arxiv:2005.08100
  · FastConformer  (NVIDIA, 2023)       arxiv:2305.05084
  · wav2vec 2.0    (Meta, 2020)         arxiv:2006.11477
  · Omnilingual ASR (Meta, 2025)        arxiv:2511.09690
  · Samba-ASR      (2025)

Datos requeridos (locales):
  /home/nmartinez-root/mi_entorno/data/common_voice_25/   (Common Voice ES)
  /home/nmartinez-root/mi_entorno/data/dave1.0/           (DAVE 1.0 ES)
  /home/nmartinez-root/mi_entorno/data/voxpopuli/         (VoxPopuli ES)

Ejecución:
  pip install -r requirements.txt
  python asr_espanol_espana.py

El script automatiza:
  1. Carga de corpus Common Voice + DAVE + VoxPopuli (local)
  2. Filtrado de hablantes peninsulares
  3. Extracción de características de audio (Mel filterbank 80 filtros)
  4. Entrenamiento del tokenizer BPE (6000 tokens, español peninsular)
  5. Entrenamiento del modelo Conformer (17 bloques, d_model=512)
  6. Decodificación (greedy + beam search beam=8)
  7. Evaluación WER/CER sobre hablantes peninsulares
"""

# ==============================================================================
# 0. IMPORTACIONES Y CONFIGURACIÓN
# ==============================================================================

import os
import sys
import math
import json
import time
import random
import logging
import pathlib
import unicodedata
import re
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ASSR")

# Rutas fijas de datos locales
DATA_BASE = pathlib.Path("/home/nmartinez-root/mi_entorno/data")
COMMON_VOICE_PATH = DATA_BASE / "common_voice_25"
DAVE_PATH = DATA_BASE / "dave1.0"
VOXPOPULI_PATH = DATA_BASE / "voxpopuli"

# ==============================================================================
# VERIFICACIÓN DE DEPENDENCIAS
# ==============================================================================

def _check_deps() -> None:
    """Verifica que las dependencias requeridas estén instaladas."""
    missing = []
    required = [
        ("torch",           "torch"),
        ("torchaudio",      "torchaudio"),
        ("numpy",           "numpy"),
        ("sentencepiece",   "sentencepiece"),
        ("datasets",        "datasets"),
        ("jiwer",           "jiwer"),
        ("librosa",         "librosa"),
        ("soundfile",       "soundfile"),
        ("requests",        "requests"),
    ]
    for import_name, install_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(install_name)
    if missing:
        log.error(
            "Dependencias faltantes. Instálalas con:\n"
            f"  pip install {' '.join(missing)}"
        )
        sys.exit(1)


def _verify_local_data() -> None:
    """Verifica que los datos locales existan."""
    log.info("\n=== VERIFICACIÓN DE DATOS LOCALES ===")
    
    if not COMMON_VOICE_PATH.exists():
        log.error(f"✗ Common Voice NO encontrado en: {COMMON_VOICE_PATH}")
        sys.exit(1)
    log.info(f"✓ Common Voice encontrado en: {COMMON_VOICE_PATH}")

    if not DAVE_PATH.exists():
        log.error(f"✗ DAVE 1.0 NO encontrado en: {DAVE_PATH}")
        sys.exit(1)
    log.info(f"✓ DAVE 1.0 encontrado en: {DAVE_PATH}")
    
    if not VOXPOPULI_PATH.exists():
        log.error(f"✗ VoxPopuli NO encontrado en: {VOXPOPULI_PATH}")
        sys.exit(1)
    log.info(f"✓ VoxPopuli encontrado en: {VOXPOPULI_PATH}")


# ==============================================================================
# FASE 1 — FRONTEND DE AUDIO
# ==============================================================================

class AudioFrontend:
    """
    Pipeline de procesamiento de audio para ASR en español peninsular.

    Pasos:
      1. Remuestreo a 16 kHz mono
      2. Normalización de amplitud (peak)
      3. VAD ajustado a patrones prosódicos del español peninsular
      4. Filtro de preénfasis (α=0.97)
      5. Ventana Hann + STFT (N=512)
      6. Banco de 80 filtros Mel
      7. Log-compresión
      8. CMVN por utterance

    Salida: tensor [T, 80]
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,       # 25 ms
        hop_length: int = 160,       # 10 ms
        pre_emphasis: float = 0.97,
        f_min: float = 20.0,
        f_max: float = 8_000.0,
        cmvn: bool = True,
        vad_energy_threshold: float = 0.01,
        vad_min_silence_ms: int = 200,
    ):
        import torch
        import torchaudio

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pre_emphasis = pre_emphasis
        self.cmvn = cmvn
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_min_silence_ms = vad_min_silence_ms

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
        )

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Carga audio y remuestrea a 16 kHz mono."""
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze().numpy(), self.sample_rate

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """Normalización de amplitud peak a 0.95."""
        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio / peak * 0.95
        return audio

    def vad_peninsular(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        VAD ajustado a patrones prosódicos del español peninsular.

        El español peninsular tiene sílabas de ~130-170 ms y pausas mínimas
        entre palabras de ~80 ms. Se usa ventana de energía de 20 ms con
        suavizado para manejar oclusivas sordas (/p/, /t/, /k/) que producen
        silencios breves dentro del habla.
        """
        frame_len = int(0.020 * sr)   # 20 ms
        min_sil_frames = max(1, int(self.vad_min_silence_ms / 20))

        n_frames = (len(audio) - frame_len) // frame_len + 1
        if n_frames <= 0:
            return audio

        energy = np.array([
            np.sum(audio[i * frame_len:(i + 1) * frame_len] ** 2)
            for i in range(n_frames)
        ])

        e_max = energy.max()
        threshold = self.vad_energy_threshold * e_max if e_max > 0 else 0.0
        speech_frames = energy > threshold

        # Suavizado: rellenar silencios cortos (pausas oclusivas, ~40-60 ms)
        for i in range(1, len(speech_frames) - 1):
            if not speech_frames[i]:
                left = max(0, i - min_sil_frames // 2)
                right = min(len(speech_frames), i + min_sil_frames // 2 + 1)
                if speech_frames[left:right].any():
                    speech_frames[i] = True

        segments = [
            audio[i * frame_len:(i + 1) * frame_len]
            for i, is_speech in enumerate(speech_frames)
            if is_speech
        ]
        return np.concatenate(segments) if segments else audio

    def pre_emphasis_filter(self, audio: np.ndarray) -> np.ndarray:
        """Filtro de preénfasis α=0.97 para realzar altas frecuencias."""
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def compute_mel_features(self, audio: np.ndarray) -> "torch.Tensor":
        """Calcula log-Mel spectrogram con CMVN por utterance."""
        import torch

        waveform = torch.FloatTensor(audio).unsqueeze(0)   # [1, T]
        mel = self.mel_transform(waveform)                 # [1, n_mels, T_frames]
        mel = torch.log(mel + 1e-6)                        # log-compresión
        mel = mel.squeeze(0).T                             # [T_frames, n_mels]

        if self.cmvn:
            mean = mel.mean(dim=0, keepdim=True)
            std = mel.std(dim=0, keepdim=True) + 1e-6
            mel = (mel - mean) / std

        return mel  # [T_frames, 80]

    def process(self, path: str) -> "torch.Tensor":
        """Pipeline completo desde fichero: ruta → tensor [T, 80]."""
        audio, sr = self.load_audio(path)
        audio = self.normalize_amplitude(audio)
        audio = self.vad_peninsular(audio, sr)
        audio = self.pre_emphasis_filter(audio)
        return self.compute_mel_features(audio)

    def process_waveform(self, waveform: np.ndarray, sr: int = 16_000) -> "torch.Tensor":
        """Pipeline completo desde waveform en memoria → tensor [T, 80]."""
        import torch
        import torchaudio

        if sr != self.sample_rate:
            wav_t = torch.FloatTensor(waveform).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, self.sample_rate)
            waveform = wav_t.squeeze().numpy()

        waveform = self.normalize_amplitude(waveform)
        waveform = self.vad_peninsular(waveform, self.sample_rate)
        waveform = self.pre_emphasis_filter(waveform)
        return self.compute_mel_features(waveform)

    @staticmethod
    def collate_batch(
        batch: List[Tuple["torch.Tensor", "torch.Tensor"]],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Collate con padding para DataLoader."""
        import torch

        features, targets = zip(*batch)
        feat_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
        tgt_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

        max_f = int(feat_lengths.max())
        max_t = int(tgt_lengths.max())
        n_mels = features[0].shape[1]
        B = len(features)

        feat_padded = torch.zeros(B, max_f, n_mels)
        tgt_padded = torch.zeros(B, max_t, dtype=torch.long)

        for i, (f, t) in enumerate(zip(features, targets)):
            feat_padded[i, :f.shape[0]] = f
            tgt_padded[i, :t.shape[0]] = t

        return feat_padded, tgt_padded, feat_lengths, tgt_lengths


# ==============================================================================
# FASE 2 — ENCODER CONFORMER
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) para codificación posicional relativa.
    Modela mejor las distancias relativas que los embeddings absolutos,
    lo que es relevante para la variabilidad prosódica del español peninsular.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def _get_freqs(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [T, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [T, dim]
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def apply(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return x * cos + self._rotate_half(x) * sin


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention con Rotary Position Embedding."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible entre num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEncoding(self.d_head)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H = self.num_heads
        dh = self.d_head

        Q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)   # [B,H,T,dh]
        K = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        cos, sin = self.rope._get_freqs(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1,1,T,dh]
        sin = sin.unsqueeze(0).unsqueeze(0)
        Q = self.rope.apply(Q, cos, sin)
        K = self.rope.apply(K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B,H,T,T]

        if key_padding_mask is not None:
            # key_padding_mask: [B, T], True = posición a ignorar
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Módulo Feed-Forward con activación Swish (SiLU)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class ConformerConvModule(nn.Module):
    """
    Módulo de convolución Conformer:
      Pointwise → GLU → Depthwise Conv → BN → Swish → Pointwise
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size debe ser impar"

        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        xn = self.norm(x).transpose(1, 2)        # [B, D, T]
        xn = self.glu(self.pw_conv1(xn))          # [B, D, T]
        xn = self.act(self.bn(self.dw_conv(xn)))  # [B, D, T]
        xn = self.dropout(self.pw_conv2(xn))      # [B, D, T]
        return xn.transpose(1, 2)                 # [B, T, D]


class ConformerBlock(nn.Module):
    """
    Bloque Conformer completo (paper Google 2020):
      FF(½) → MHSA_RoPE → Conv → FF(½) → LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttentionRoPE(d_model, num_heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), key_padding_mask))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConvSubsampling(nn.Module):
    """
    Submuestreo CNN 8x mediante 3 × Conv2d(stride=2).
      Entrada : [B, T,   n_mels]
      Salida  : [B, T/8, d_model]
    """

    def __init__(self, n_mels: int = 80, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Calcular dimensión de salida dinámicamente
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, n_mels)
            out = self.conv(dummy)
            linear_in = out.shape[1] * out.shape[3]

        self.proj = nn.Linear(linear_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T, F = x.shape
        x = x.unsqueeze(1)                               # [B, 1, T, F]
        x = self.conv(x)                                 # [B, C, T', F']
        B2, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B2, T2, -1)  # [B, T', C*F']
        x = self.dropout(self.proj(x))                  # [B, T', d_model]
        return x, T2


class ConformerEncoder(nn.Module):
    """
    Encoder Conformer (FastConformer-style):
      · Submuestreo CNN 8x
      · Proyección lineal 80 → d_model=512
      · 17 bloques Conformer con RoPE
      · LayerNorm final
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        num_layers: int = 17,
        num_heads: int = 8,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d_model, dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
          x             : [B, T, n_mels]
          input_lengths : [B] longitudes reales antes del padding

        Returns:
          encoder_out       : [B, T', d_model]
          out_lengths       : [B] longitudes tras el submuestreo
          key_padding_mask  : [B, T'] máscara de padding
        """
        x, T2 = self.subsampling(x)       # [B, T', d_model]

        out_lengths = None
        key_padding_mask = None
        if input_lengths is not None:
            out_lengths = (input_lengths // 8).clamp(min=1)
            B, max_len = x.shape[0], x.shape[1]
            key_padding_mask = (
                torch.arange(max_len, device=x.device).unsqueeze(0)
                >= out_lengths.unsqueeze(1)
            )

        for block in self.blocks:
            x = block(x, key_padding_mask)

        return self.final_norm(x), out_lengths, key_padding_mask


# ==============================================================================
# FASE 3 — TOKENIZACIÓN ESPAÑOL PENINSULAR
# ==============================================================================

class SpanishTextNormalizer:
    """
    Normalización de texto para el español peninsular.

    Conserva: ñ, acentos (á é í ó ú), ü, dígrafos ll / ch / rr
    Distinción s/z/c (sin ceceo ni seseo)
    Distinción ll/y
    """

    VALID_CHARS = set(
        "abcdefghijklmnñopqrstuvwxyz"
        "áéíóúüý"
        "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
        "ÁÉÍÓÚÜÝ"
        " \t\n.,;:!?¡¿()'\"%-"
        "0123456789"
    )

    def normalize(self, text: str) -> str:
        """Normalización NFC preservando características del español peninsular."""
        text = unicodedata.normalize("NFC", text)
        # Eliminar caracteres de control
        text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
        # Normalizar espacios
        text = re.sub(r"\s+", " ", text).strip()
        # Filtrar a caracteres válidos
        text = "".join(c for c in text if c in self.VALID_CHARS)
        return text

    def for_tokenizer(self, text: str) -> str:
        """Prepara texto para entrenamiento del tokenizer (minúsculas, sin números)."""
        text = self.normalize(text)
        text = text.lower()
        text = re.sub(r"\d+", " <num> ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class SpanishBPETokenizer:
    """
    Tokenizador BPE para español peninsular.

    Vocabulario: 6000 tokens
    Tokens especiales: <blank>(0), <sos>(1), <eos>(2), <unk>(3), <num>(4), <sil>(5)
    Entrenado sobre texto peninsular (noticias, BOE, subtítulos RTVE)
    """

    SPECIAL_TOKENS = ["<blank>", "<sos>", "<eos>", "<unk>", "<num>", "<sil>"]
    BLANK_ID = 0
    SOS_ID   = 1
    EOS_ID   = 2
    UNK_ID   = 3
    NUM_ID   = 4
    SIL_ID   = 5

    def __init__(
        self,
        vocab_size: int = 6000,
        model_dir: str = "tokenizer",
    ) -> None:
        self.vocab_size = vocab_size
        self.model_dir = pathlib.Path(model_dir)
        self.model_path = self.model_dir / "es_peninsular.model"
        self.normalizer = SpanishTextNormalizer()
        self.sp = None

    def train(self, texts: List[str]) -> None:
        """
        Entrena tokenizer SentencePiece BPE sobre corpus peninsular.

        El corpus debe contener texto normalizado de fuentes peninsulares:
        noticias (El País, El Mundo), BOE, subtítulos RTVE.
        """
        import sentencepiece as spm

        self.model_dir.mkdir(parents=True, exist_ok=True)
        corpus_file = self.model_dir / "corpus_peninsular.txt"

        log.info(f"Preparando corpus: {len(texts):,} textos...")
        normalized = [self.normalizer.for_tokenizer(t) for t in texts if t and t.strip()]
        corpus_file.write_text("\n".join(normalized), encoding="utf-8")

        model_prefix = str(self.model_path.with_suffix(""))
        user_symbols = ",".join(["<blank>", "<num>", "<sil>"])

        PENINSULAR_CHARS = "ñáéíóúüÑÁÉÍÓÚÜ"
        all_text = " ".join(normalized)
        present_required = "".join(c for c in PENINSULAR_CHARS if c in all_text)

        total_chars = len(all_text)
        unique_chars = len(set(all_text) - {" "})
        unique_bigrams = len({w[i:i+2] for w in all_text.split() for i in range(len(w) - 1)})

        if total_chars >= 1_000_000:
            effective_vocab = self.vocab_size
        elif total_chars >= 100_000:
            effective_vocab = min(self.vocab_size, unique_chars + unique_bigrams * 5)
        else:
            effective_vocab = min(self.vocab_size, max(200, unique_chars + unique_bigrams))

        if effective_vocab < self.vocab_size:
            log.warning(
                f"Corpus pequeño ({total_chars:,} chars). "
                f"Reduciendo vocab_size {self.vocab_size} → {effective_vocab}. "
                f"Con el corpus real (Common Voice + VoxPopuli, >1M chars) "
                f"se alcanzará vocab={self.vocab_size}."
            )

        log.info(f"Entrenando BPE vocab={effective_vocab}...")
        train_kwargs: Dict[str, Any] = dict(
            input=str(corpus_file),
            model_prefix=model_prefix,
            vocab_size=effective_vocab,
            model_type="bpe",
            character_coverage=0.9995,
            user_defined_symbols=user_symbols,
            pad_id=-1,
            unk_id=self.UNK_ID,
            bos_id=self.SOS_ID,
            eos_id=self.EOS_ID,
            normalization_rule_name="nmt_nfkc",
            split_digits=False,
            split_by_unicode_script=False,
            hard_vocab_limit=False,
        )
        if present_required:
            train_kwargs["required_chars"] = present_required

        spm.SentencePieceTrainer.train(**train_kwargs)
        log.info(f"Tokenizer guardado: {self.model_path}")
        self._load(str(self.model_path))

    def _load(self, path: str) -> None:
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path)
        self.model_path = pathlib.Path(path)

    def load(self) -> None:
        """Carga modelo guardado desde model_dir."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        self._load(str(self.model_path))

    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Texto → lista de IDs."""
        if self.sp is None:
            raise RuntimeError("Tokenizer no inicializado. Llama a train() o load().")
        text = self.normalizer.normalize(text)
        ids: List[int] = self.sp.Encode(text, out_type=int)
        if add_sos:
            ids = [self.SOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Lista de IDs → texto."""
        if self.sp is None:
            raise RuntimeError("Tokenizer no inicializado.")
        if skip_special:
            special = {self.BLANK_ID, self.SOS_ID, self.EOS_ID, self.SIL_ID}
            ids = [i for i in ids if i not in special]
        return self.sp.Decode(ids)

    def vocab_size_real(self) -> int:
        return self.sp.GetPieceSize() if self.sp is not None else self.vocab_size

    @property
    def is_trained(self) -> bool:
        return self.sp is not None or self.model_path.exists()


# ==============================================================================
# FASE 4 — DECODER: CTC + AED TRANSFORMER
# ==============================================================================

class CTCDecoder(nn.Module):
    """
    Decoder CTC principal.
    Rápido en inferencia; usado como decoder principal (peso 0.3 en el loss).
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(self.norm(x)))


class AEDDecoderLayer(nn.Module):
    """
    Capa del decoder AED.
    La cross-attention atiende a las terminaciones verbales del español
    peninsular (-áis, -éis, -ción, -mente) a través de los pesos de atención.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_drop = nn.Dropout(dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_drop = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention causal
        t = self.self_attn_norm(tgt)
        t_sa, _ = self.self_attn(t, t, t, attn_mask=tgt_mask)
        tgt = tgt + self.self_attn_drop(t_sa)

        # Cross-attention sobre encoder
        t = self.cross_attn_norm(tgt)
        t_ca, _ = self.cross_attn(
            t, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.cross_attn_drop(t_ca)

        # Feed-forward
        tgt = tgt + self.ff(self.ff_norm(tgt))
        return tgt


class AEDDecoder(nn.Module):
    """
    Decoder AED Transformer (4 capas, peso 0.7 en el loss).
    Decoder secundario con mayor capacidad lingüística.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            AEDDecoderLayer(d_model, num_heads, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.embed_drop(
            self.embed(tgt_ids) * math.sqrt(self.d_model)
            + self.pos_embed(positions)
        )

        # Máscara causal (float para PyTorch MHA)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        causal_f = causal.masked_fill(causal.bool(), float("-inf"))

        for layer in self.layers:
            x = layer(
                x, memory,
                tgt_mask=causal_f,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.proj(self.norm(x))   # [B, T, vocab_size]


# ==============================================================================
# MODELO ASSR COMPLETO
# ==============================================================================

class ASSR(nn.Module):
    """
    ASSR — Automatic Speech Recognition para español de España.

    Encoder : Conformer (17 bloques, d_model=512, RoPE, subsampling 8x)
    Decoder1: CTC  — decoder principal  (velocidad)
    Decoder2: AED  — decoder secundario (precisión)
    Loss    : 0.3 · L_CTC + 0.7 · L_CE
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 17,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        ctc_weight: float = 0.3,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight

        self.encoder = ConformerEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            kernel_size=conv_kernel_size,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )
        self.ctc_decoder = CTCDecoder(d_model, vocab_size, dropout)
        self.aed_decoder = AEDDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_out, out_lengths, enc_mask = self.encoder(features, input_lengths)
        ctc_logits = self.ctc_decoder(enc_out)

        result: Dict[str, Any] = {
            "encoder_out": enc_out,
            "ctc_logits": ctc_logits,
            "out_lengths": out_lengths,
        }

        if targets is not None:
            tgt_in = targets[:, :-1]   # [<sos>, t1, ..., t_{n-1}]
            result["aed_logits"] = self.aed_decoder(tgt_in, enc_out, enc_mask)

        return result

    def compute_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = self.forward(features, input_lengths, targets)

        ctc_logits = result["ctc_logits"]
        aed_logits = result["aed_logits"]
        out_lengths = result["out_lengths"]

        # CTC loss
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
        ctc_loss = F.ctc_loss(
            log_probs, targets, out_lengths, target_lengths,
            blank=0, zero_infinity=True,
        )

        # AED cross-entropy loss: predice targets[1:] desde targets[:-1]
        tgt_out = targets[:, 1:]                    # [B, T_out]
        B, T_out, V = aed_logits.shape
        ce_loss = F.cross_entropy(
            aed_logits.reshape(-1, V),
            tgt_out.reshape(-1),
            ignore_index=0,
        )

        total = self.ctc_weight * ctc_loss + (1.0 - self.ctc_weight) * ce_loss
        return total, {
            "loss": total.item(),
            "ctc_loss": ctc_loss.item(),
            "ce_loss": ce_loss.item(),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# FASE 5 — ENTRENAMIENTO
# ==============================================================================

def spec_augment(
    features: torch.Tensor,
    num_time_masks: int = 2,
    max_time_mask_frac: float = 0.10,   # ~10% de T (sílabas más largas en español)
    num_freq_masks: int = 2,
    max_freq_mask: int = 27,
) -> torch.Tensor:
    """
    SpecAugment con máscaras adaptadas al español peninsular.

    Las sílabas del español (~130-170 ms) son más largas que las del inglés
    (~80 ms), por lo que se usa una fracción relativa del tiempo en vez de
    un valor absoluto de frames.
    """
    B, T, F = features.shape
    aug = features.clone()
    t_max = max(1, int(T * max_time_mask_frac))

    for i in range(B):
        for _ in range(num_time_masks):
            t = random.randint(0, t_max)
            t0 = random.randint(0, max(0, T - t))
            aug[i, t0:t0 + t, :] = 0.0
        for _ in range(num_freq_masks):
            f = random.randint(0, max_freq_mask)
            f0 = random.randint(0, max(0, F - f))
            aug[i, :, f0:f0 + f] = 0.0

    return aug


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """AdamW + warmup lineal 10k pasos + cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: ASSR,
    loader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    use_spec_augment: bool = True,
    max_grad_norm: float = 1.0,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Entrena una época."""
    model.train()
    total_loss = total_ctc = total_ce = 0.0
    n = 0

    for idx, (features, targets, feat_lengths, tgt_lengths) in enumerate(loader):
        features    = features.to(device)
        targets     = targets.to(device)
        feat_lengths = feat_lengths.to(device)
        tgt_lengths  = tgt_lengths.to(device)

        if use_spec_augment:
            features = spec_augment(features)

        if idx == 0:
            log.info(
                f"DEBUG - Batch shapes: features={tuple(features.shape)}, "
                f"targets={tuple(targets.shape)}"
            )
            log.info(
                f"DEBUG - Feature stats: min={features.min():.3f}, "
                f"max={features.max():.3f}, mean={features.mean():.3f}"
            )
            log.info(
                f"DEBUG - Target sample (primeros 20 IDs): "
                f"{targets[0, :20].tolist()}"
            )

        optimizer.zero_grad()
        loss, metrics = model.compute_loss(features, targets, feat_lengths, tgt_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += metrics["loss"]
            total_ctc  += metrics["ctc_loss"]
            total_ce   += metrics["ce_loss"]
            n += 1

        if idx % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            log.info(
                f"  [{idx:5d}/{len(loader):5d}] "
                f"loss={metrics['loss']:.4f}  ctc={metrics['ctc_loss']:.4f}  "
                f"ce={metrics['ce_loss']:.4f}  lr={lr:.2e}"
            )

    denom = max(1, n)
    return {"loss": total_loss / denom, "ctc_loss": total_ctc / denom, "ce_loss": total_ce / denom}


@torch.no_grad()
def validate_epoch(
    model: ASSR,
    loader: "DataLoader",
    device: torch.device,
) -> Dict[str, float]:
    """Valida el modelo."""
    model.eval()
    total_loss = 0.0
    n = 0

    for features, targets, feat_lengths, tgt_lengths in loader:
        features     = features.to(device)
        targets      = targets.to(device)
        feat_lengths = feat_lengths.to(device)
        tgt_lengths  = tgt_lengths.to(device)

        loss, metrics = model.compute_loss(features, targets, feat_lengths, tgt_lengths)
        if not torch.isnan(loss):
            total_loss += metrics["loss"]
            n += 1

    return {"val_loss": total_loss / max(1, n)}


def train(
    model: ASSR,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    num_epochs: int = 50,
    learning_rate: float = 5e-4,
    warmup_steps: int = 10_000,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
) -> None:
    """Loop principal de entrenamiento con checkpointing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    ckpt_dir = pathlib.Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=1e-2,
    )
    total_steps = num_epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from and pathlib.Path(resume_from).exists():
        log.info(f"Reanudando entrenamiento desde {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    log.info(f"Entrenamiento iniciado — {model.num_parameters():,} parámetros")
    log.info(f"Device: {device}  |  Épocas: {num_epochs}  |  LR: {learning_rate}")

    for epoch in range(start_epoch, num_epochs):
        log.info(f"\n{'='*60}\nÉPOCA {epoch + 1}/{num_epochs}\n{'='*60}")
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics   = validate_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        log.info(
            f"Época {epoch+1} | train_loss={train_metrics['loss']:.4f} "
            f"| val_loss={val_metrics['val_loss']:.4f} | {elapsed:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            log.info(f"✓ Nuevo mejor modelo guardado (val_loss={best_val_loss:.4f})")

    log.info(f"\nEntrenamiento completo. Mejor val_loss: {best_val_loss:.4f}")


# ==============================================================================
# FASE 6 — DATASETS ESPECÍFICOS DE ESPAÑA
# ==============================================================================

# Acentos/regiones peninsulares (incluir)
PENINSULAR_ACCENTS = {
    "spain", "españa", "castellano", "castilla", "madrid", "barcelona",
    "valencia", "sevilla", "bilbao", "zaragoza", "galicia", "asturias",
    "murcia", "extremadura", "aragón", "aragon", "la rioja", "cantabria",
    "navarra", "castilla y león", "castilla-la mancha", "comunidad valenciana",
    "país vasco", "cataluña", "cataluna", "baleares", "canarias",
}

# Variantes latinoamericanas (excluir)
EXCLUDED_ACCENTS = {
    "mexico", "méxico", "argentina", "colombia", "chile", "peru", "perú",
    "venezuela", "ecuador", "bolivia", "paraguay", "uruguay", "cuba",
    "dominican republic", "república dominicana", "puerto rico",
    "costa rica", "guatemala", "honduras", "nicaragua", "el salvador",
    "panamá", "panama", "latin american", "latinoamérica",
}


def _is_peninsular(sample: Dict) -> bool:
    """Devuelve True si la muestra es de un hablante peninsular o desconocido."""
    accent = (sample.get("accent") or "").lower().strip()
    if not accent:
        return True  # sin información → incluir
    for exc in EXCLUDED_ACCENTS:
        if exc in accent:
            return False
    # Si hay info explícita de España → incluir
    for pen in PENINSULAR_ACCENTS:
        if pen in accent:
            return True
    # Acento no identificado como latinoamericano → incluir por defecto
    return True


def load_local_datasets() -> Dict[str, Any]:
    """
    Carga Common Voice ES, DAVE 1.0 ES y VoxPopuli ES desde rutas locales.
    
    Returns:
        Dict con keys "common_voice", "dave" y "voxpopuli", cada uno conteniendo
        un dataset de HuggingFace con splits train/validation/test
    """
    from datasets import load_from_disk
    
    datasets_loaded: Dict[str, Any] = {}
    
    # ── Common Voice ───────────────────────────────────────────────────────
    log.info(f"Cargando Common Voice ES desde: {COMMON_VOICE_PATH}")
    try:
        # Intentar cargarlo directamente si está guardado como dataset local
        cv_dataset = load_from_disk(str(COMMON_VOICE_PATH))
        log.info(f"✓ Common Voice cargado con splits: {list(cv_dataset.keys())}")
        datasets_loaded["common_voice"] = cv_dataset
    except Exception as exc:
        log.warning(f"No se pudo cargar Common Voice como dataset local: {exc}")
        # Intentar cargarlo desde HuggingFace cache
        try:
            from datasets import load_dataset
            cv_dataset = load_dataset(
                "mozilla-foundation/common_voice_16_0",
                "es",
                cache_dir=str(COMMON_VOICE_PATH / "hf_cache"),
                trust_remote_code=True,
            )
            log.info(f"✓ Common Voice cargado desde HF cache con splits: {list(cv_dataset.keys())}")
            datasets_loaded["common_voice"] = cv_dataset
        except Exception as exc2:
            log.error(f"No se pudo cargar Common Voice: {exc2}")

    # ── DAVE 1.0 ────────────────────────────────────────────────────────────
    log.info(f"Cargando DAVE 1.0 ES desde: {DAVE_PATH}")
    try:
        dave_dataset = load_from_disk(str(DAVE_PATH))
        log.info(f"✓ DAVE 1.0 cargado con splits: {list(dave_dataset.keys())}")
        datasets_loaded["dave"] = dave_dataset
    except Exception as exc:
        log.error(f"No se pudo cargar DAVE 1.0: {exc}")
    
    # ── VoxPopuli ──────────────────────────────────────────────────────────
    log.info(f"Cargando VoxPopuli ES desde: {VOXPOPULI_PATH}")
    try:
        vp_dataset = load_from_disk(str(VOXPOPULI_PATH))
        log.info(f"✓ VoxPopuli cargado con splits: {list(vp_dataset.keys())}")
        datasets_loaded["voxpopuli"] = vp_dataset
    except Exception as exc:
        log.warning(f"No se pudo cargar VoxPopuli como dataset local: {exc}")
        # Intentar cargarlo desde HuggingFace cache
        try:
            from datasets import load_dataset
            vp_dataset = load_dataset(
                "facebook/voxpopuli",
                "es",
                cache_dir=str(VOXPOPULI_PATH / "hf_cache"),
                trust_remote_code=True,
            )
            log.info(f"✓ VoxPopuli cargado desde HF cache con splits: {list(vp_dataset.keys())}")
            datasets_loaded["voxpopuli"] = vp_dataset
        except Exception as exc2:
            log.error(f"No se pudo cargar VoxPopuli: {exc2}")
    
    if not datasets_loaded:
        log.error("No se pudo cargar ningún dataset local")
        sys.exit(1)
    
    return datasets_loaded


def _hf_to_samples(hf_dataset: Any, split: str, source: str) -> List[Dict]:
    """Convierte un split de HuggingFace Dataset a lista de dicts."""
    samples = []
    if hf_dataset is None or split not in hf_dataset:
        return samples
    for row in hf_dataset[split]:
        # Common Voice/DAVE/VoxPopuli: posibles nombres de campo de texto
        text = row.get("sentence") or row.get("raw_text") or row.get("text") or ""
        if not text.strip():
            continue
        samples.append({
            "audio": row.get("audio", {}),
            "sentence": text,
            "accent": row.get("accent", ""),
            "source": source,
            "split": split,
        })
    return samples


class SpanishASRDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch para ASR en español peninsular.

    Compatible con Common Voice ES y VoxPopuli ES.
    Soporta speed perturbation: 0.9×, 1.0×, 1.1×.
    """

    def __init__(
        self,
        samples: List[Dict],
        audio_frontend: AudioFrontend,
        tokenizer: SpanishBPETokenizer,
        speed_perturbation: bool = False,
        speed_rates: Tuple[float, ...] = (0.9, 1.0, 1.1),
        max_audio_sec: float = 30.0,
        max_tokens: int = 400,
    ) -> None:
        self.samples = samples
        self.frontend = audio_frontend
        self.tokenizer = tokenizer
        self.speed_perturbation = speed_perturbation
        self.speed_rates = speed_rates
        sr = audio_frontend.sample_rate
        self.max_frames = int(max_audio_sec * sr / audio_frontend.hop_length)
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # ── Obtener waveform ──────────────────────────────────────────────
        waveform: Optional[np.ndarray] = None
        sr = 16_000

        audio_field = sample.get("audio") or {}
        if isinstance(audio_field, dict) and audio_field.get("array") is not None:
            waveform = np.array(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", 16_000))
        elif isinstance(audio_field, dict) and audio_field.get("path"):
            audio_path = pathlib.Path(audio_field["path"])
            if audio_path.exists():
                features = self.frontend.process(str(audio_path))
                text = sample.get("sentence") or sample.get("text") or ""
                tokens = torch.tensor(
                    self.tokenizer.encode(text, add_sos=True, add_eos=True),
                    dtype=torch.long,
                )
                return features[: self.max_frames], tokens[: self.max_tokens]
        elif sample.get("array") is not None:
            waveform = np.array(sample["array"], dtype=np.float32)
            sr = int(sample.get("sampling_rate", 16_000))
        elif sample.get("path") and pathlib.Path(sample["path"]).exists():
            features = self.frontend.process(sample["path"])
            text = sample.get("sentence") or sample.get("text") or ""
            tokens = torch.tensor(
                self.tokenizer.encode(text, add_sos=True, add_eos=True),
                dtype=torch.long,
            )
            return features[: self.max_frames], tokens[: self.max_tokens]

        if waveform is None or len(waveform) == 0:
            raise ValueError("No se pudo cargar audio real para la muestra.")

        # Speed perturbation
        if self.speed_perturbation and random.random() < 0.67:
            try:
                import librosa
                rate = random.choice(self.speed_rates)
                if rate != 1.0:
                    waveform = librosa.effects.time_stretch(waveform, rate=rate)
            except Exception:
                pass

        features = self.frontend.process_waveform(waveform, sr)

        text = sample.get("sentence") or sample.get("text") or ""
        tokens = torch.tensor(
            self.tokenizer.encode(text, add_sos=True, add_eos=True),
            dtype=torch.long,
        )
        return features[: self.max_frames], tokens[: self.max_tokens]

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padding por duración (~200 s de audio por batch)."""
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        features, targets = zip(*batch)

        feat_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
        tgt_lengths  = torch.tensor([t.shape[0] for t in targets],  dtype=torch.long)

        n_mels   = features[0].shape[1] if features[0].dim() > 1 else 80
        max_feat = int(feat_lengths.max())
        max_tgt  = int(tgt_lengths.max())
        B = len(features)

        feat_pad = torch.zeros(B, max_feat, n_mels)
        tgt_pad  = torch.zeros(B, max_tgt, dtype=torch.long)

        for i, (f, t) in enumerate(zip(features, targets)):
            feat_pad[i, : f.shape[0]] = f
            tgt_pad[i, : t.shape[0]] = t

        return feat_pad, tgt_pad, feat_lengths, tgt_lengths


def build_dataloaders(
    tokenizer: SpanishBPETokenizer,
    audio_frontend: AudioFrontend,
    batch_size: int = 8,
    num_workers: int = 0,
    val_fraction: float = 0.05,
) -> Tuple["DataLoader", "DataLoader", List[str]]:
    """
    Construye los DataLoaders de entrenamiento y validación usando
    Common Voice ES + DAVE 1.0 ES + VoxPopuli ES desde rutas locales.

    Devuelve también la lista de textos del corpus para entrenar el tokenizer.
    """
    from torch.utils.data import DataLoader

    all_samples: List[Dict] = []
    corpus_texts: List[str] = []

    # ── Cargar datasets locales ────────────────────────────────────────────
    datasets = load_local_datasets()

    # ── Common Voice ES ────────────────────────────────────────────────────
    if "common_voice" in datasets:
        cv = datasets["common_voice"]
        log.info("Procesando Common Voice ES...")
        for split in ("train", "validation", "test"):
            ss = _hf_to_samples(cv, split, "common_voice")
            # Filtrar solo hablantes peninsulares
            ss_filtered = [s for s in ss if _is_peninsular(s)]
            all_samples.extend(ss_filtered)
            corpus_texts.extend(s["sentence"] for s in ss_filtered)
            log.info(f"  Common Voice '{split}': {len(ss_filtered):,} muestras peninsulares")

    # ── VoxPopuli ES ───────────────────────────────────────────────────────
    if "voxpopuli" in datasets:
        vp = datasets["voxpopuli"]
        log.info("Procesando VoxPopuli ES...")
        for split in ("train", "validation", "test"):
            ss = _hf_to_samples(vp, split, "voxpopuli")
            # VoxPopuli es principalmente de fuentes públicas españolas
            ss_filtered = [s for s in ss if _is_peninsular(s)]
            all_samples.extend(ss_filtered)
            corpus_texts.extend(s["sentence"] for s in ss_filtered)
            log.info(f"  VoxPopuli '{split}': {len(ss_filtered):,} muestras")

    # ── DAVE 1.0 ES ─────────────────────────────────────────────────────────
    if "dave" in datasets:
        dave = datasets["dave"]
        log.info("Procesando DAVE 1.0 ES...")
        for split in ("train", "validation", "test"):
            ss = _hf_to_samples(dave, split, "dave1.0")
            ss_filtered = [s for s in ss if _is_peninsular(s)]
            all_samples.extend(ss_filtered)
            corpus_texts.extend(s["sentence"] for s in ss_filtered)
            log.info(f"  DAVE 1.0 '{split}': {len(ss_filtered):,} muestras")

    if not all_samples:
        log.error("No se encontraron muestras reales en Common Voice, DAVE o VoxPopuli.")
        sys.exit(1)

    # ── Split train/val ────────────────────────────────────────────────────
    random.shuffle(all_samples)
    n_val = max(50, int(len(all_samples) * val_fraction))
    train_samples = all_samples[n_val:]
    val_samples   = all_samples[:n_val]
    log.info(f"Train: {len(train_samples):,}  |  Val: {len(val_samples):,}")

    train_ds = SpanishASRDataset(
        train_samples, audio_frontend, tokenizer, speed_perturbation=True
    )
    val_ds = SpanishASRDataset(
        val_samples, audio_frontend, tokenizer, speed_perturbation=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=SpanishASRDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SpanishASRDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, corpus_texts


# ==============================================================================
# FASE 7 — DECODIFICACIÓN
# ==============================================================================

@torch.no_grad()
def greedy_ctc_decode(
    model: ASSR,
    features: torch.Tensor,
    tokenizer: SpanishBPETokenizer,
    device: Optional[torch.device] = None,
) -> str:
    """Decodificación greedy CTC (baseline rápido)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    feat = features.unsqueeze(0).to(device) if features.dim() == 2 else features.to(device)
    result = model(feat)
    ids = result["ctc_logits"].argmax(dim=-1).squeeze(0).tolist()

    # Eliminar blanks (id=0) y duplicados consecutivos
    decoded: List[int] = []
    prev = -1
    for idx in ids:
        if idx != prev and idx != 0:
            decoded.append(idx)
        prev = idx

    return tokenizer.decode(decoded)


@torch.no_grad()
def beam_search_ctc_decode(
    model: ASSR,
    features: torch.Tensor,
    tokenizer: SpanishBPETokenizer,
    beam_size: int = 8,
    device: Optional[torch.device] = None,
) -> str:
    """
    Beam search CTC con beam=8.
    Compatible con KenLM si está instalado (rescoring externo).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    feat = features.unsqueeze(0).to(device) if features.dim() == 2 else features.to(device)
    result = model(feat)
    log_probs = F.log_softmax(result["ctc_logits"].squeeze(0), dim=-1)   # [T, V]
    T, V = log_probs.shape

    # Beam: list of (log_prob, sequence_tokens)
    beams: List[Tuple[float, List[int]]] = [(0.0, [])]

    for t in range(T):
        step_lp = log_probs[t]
        topk_scores, topk_ids = step_lp.topk(min(beam_size * 2, V))
        cands: Dict[Tuple[int, ...], float] = {}

        for beam_score, beam_seq in beams:
            for score, tok in zip(topk_scores.tolist(), topk_ids.tolist()):
                new_score = beam_score + score
                if tok == 0:                               # blank
                    key = tuple(beam_seq)
                elif beam_seq and beam_seq[-1] == tok:     # duplicado
                    key = tuple(beam_seq)
                else:
                    key = tuple(beam_seq + [tok])
                if key not in cands or cands[key] < new_score:
                    cands[key] = new_score

        sorted_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:beam_size]
        beams = [(score, list(seq)) for seq, score in sorted_cands]

    if not beams:
        return ""
    return tokenizer.decode(beams[0][1])


@torch.no_grad()
def aed_decode(
    model: ASSR,
    features: torch.Tensor,
    tokenizer: SpanishBPETokenizer,
    max_len: int = 200,
    device: Optional[torch.device] = None,
) -> str:
    """Decodificación autoregresiva con el decoder AED."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    feat = features.unsqueeze(0).to(device) if features.dim() == 2 else features.to(device)
    enc_out, _, enc_mask = model.encoder(feat)

    generated = [tokenizer.SOS_ID]
    for _ in range(max_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model.aed_decoder(tgt, enc_out, enc_mask)
        next_tok = int(logits[0, -1].argmax().item())
        if next_tok == tokenizer.EOS_ID:
            break
        generated.append(next_tok)

    return tokenizer.decode(generated[1:])   # excluir <sos>


# ==============================================================================
# FASE 8 — EVALUACIÓN
# ==============================================================================

def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """Word Error Rate (WER) sobre hablantes peninsulares."""
    try:
        import jiwer
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
        ])
        wer = jiwer.wer(
            references, hypotheses,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )
        return float(wer)
    except Exception as exc:
        log.warning(f"jiwer no disponible: {exc}. Usando WER simple.")
        return _simple_wer(hypotheses, references)


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """Character Error Rate (CER) sobre hablantes peninsulares."""
    try:
        import jiwer
        cer = jiwer.cer(references, hypotheses)
        return float(cer)
    except Exception as exc:
        log.warning(f"jiwer no disponible: {exc}. Usando CER simple.")
        return _simple_cer(hypotheses, references)


def _simple_wer(hyps: List[str], refs: List[str]) -> float:
    """WER manual (sin jiwer)."""
    total_errors = total_words = 0
    for h, r in zip(hyps, refs):
        h_words = h.lower().split()
        r_words = r.lower().split()
        total_words += len(r_words)
        total_errors += _edit_distance(h_words, r_words)
    return total_errors / max(1, total_words)


def _simple_cer(hyps: List[str], refs: List[str]) -> float:
    """CER manual (sin jiwer)."""
    total_errors = total_chars = 0
    for h, r in zip(hyps, refs):
        total_chars += len(r)
        total_errors += _edit_distance(list(h.lower()), list(r.lower()))
    return total_errors / max(1, total_chars)


def _edit_distance(a: List, b: List) -> int:
    """Distancia de edición (Levenshtein)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _analyse_peninsular_features(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Análisis específico del español peninsular:
      - Distinción c/z (ceceo → error en distinción)
      - Formas de vosotros (-áis, -éis)
      - Léxico peninsular (vosotros, coger, ordenador, móvil...)
    """
    ceceo_errors = 0
    ceceo_total = 0
    vosotros_errors = 0
    vosotros_total = 0

    vosotros_forms = re.compile(r"\b(vosotros|vosotras|[a-záéíóú]+[áé]is)\b")
    distinction_pairs = [("caza", "casa"), ("cena", "sena"), ("zumo", "sumo"),
                         ("cocer", "coser"), ("cima", "sima")]

    for hyp, ref in zip(hyps, refs):
        hyp_l = hyp.lower()
        ref_l = ref.lower()

        # Análisis de vosotros
        ref_matches = vosotros_forms.findall(ref_l)
        if ref_matches:
            vosotros_total += len(ref_matches)
            hyp_matches = vosotros_forms.findall(hyp_l)
            vosotros_errors += len(ref_matches) - len(hyp_matches)

        # Análisis distinción s/z/c
        for pair in distinction_pairs:
            if pair[0] in ref_l:
                ceceo_total += 1
                if pair[1] in hyp_l and pair[0] not in hyp_l:
                    ceceo_errors += 1

    return {
        "distincion_cz_errors": ceceo_errors / max(1, ceceo_total),
        "vosotros_recall": 1.0 - vosotros_errors / max(1, vosotros_total),
    }


@torch.no_grad()
def evaluate(
    model: ASSR,
    loader: "DataLoader",
    tokenizer: SpanishBPETokenizer,
    device: Optional[torch.device] = None,
    decode_mode: str = "beam",      # "greedy" | "beam" | "aed"
    beam_size: int = 8,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluación completa sobre hablantes peninsulares.

    Métricas:
      - WER (Word Error Rate)
      - CER (Character Error Rate)
      - Distinción c/z peninsular
      - Recuperación de formas de vosotros
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_hyps: List[str] = []
    all_refs: List[str] = []

    for batch_idx, (features, targets, feat_lengths, _) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        for i in range(features.shape[0]):
            feat_i = features[i, : feat_lengths[i]]

            if decode_mode == "greedy":
                hyp = greedy_ctc_decode(model, feat_i, tokenizer, device)
            elif decode_mode == "aed":
                hyp = aed_decode(model, feat_i, tokenizer, device=device)
            else:
                hyp = beam_search_ctc_decode(model, feat_i, tokenizer, beam_size, device)

            ref_ids = targets[i].tolist()
            ref = tokenizer.decode([x for x in ref_ids if x > 0])

            all_hyps.append(hyp)
            all_refs.append(ref)

    wer = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)
    peninsular = _analyse_peninsular_features(all_hyps, all_refs)

    metrics = {"wer": wer, "cer": cer, **peninsular}
    log.info(
        f"Evaluación ({decode_mode}) | "
        f"WER={wer:.4f} ({wer*100:.2f}%)  "
        f"CER={cer:.4f} ({cer*100:.2f}%)  "
        f"distincion_cz_err={peninsular['distincion_cz_errors']:.4f}  "
        f"vosotros_recall={peninsular['vosotros_recall']:.4f}"
    )
    return metrics


# ==============================================================================
# FUNCIÓN PRINCIPAL — PIPELINE COMPLETO
# ==============================================================================

def main() -> None:
    """
    Ejecuta el pipeline completo:
      1. Verifica que los datos locales existan
      2. Entrena tokenizer BPE peninsular
      3. Construye DataLoaders desde rutas locales
      4. Inicializa modelo ASSR Conformer
      5. Entrena
      6. Evalúa (WER/CER)
      7. Guarda modelo y resultados
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ASSR Español de España — Pipeline completo"
    )
    parser.add_argument("--output-dir",     default="output",      help="Directorio de salida")
    parser.add_argument("--epochs",         type=int,  default=50, help="Número de épocas")
    parser.add_argument("--batch-size",     type=int,  default=8,  help="Tamaño de batch")
    parser.add_argument("--lr",             type=float,default=5e-4,help="Learning rate")
    parser.add_argument("--warmup-steps",   type=int,  default=10_000, help="Pasos de warmup")
    parser.add_argument("--d-model",        type=int,  default=512, help="Dimensión del modelo")
    parser.add_argument("--encoder-layers", type=int,  default=17, help="Bloques Conformer")
    parser.add_argument("--decoder-layers", type=int,  default=4,  help="Capas AED")
    parser.add_argument("--num-heads",      type=int,  default=8,  help="Cabezas de atención")
    parser.add_argument("--vocab-size",     type=int,  default=6000, help="Vocabulario BPE")
    parser.add_argument("--dropout",        type=float,default=0.1, help="Dropout")
    parser.add_argument("--num-workers",    type=int,  default=0,  help="Workers DataLoader")
    parser.add_argument("--device",         default=None, help="cuda | cpu (auto si None)")
    parser.add_argument("--resume",         default=None, help="Checkpoint para reanudar")
    parser.add_argument("--eval-only",      action="store_true",   help="Solo evaluar")
    parser.add_argument("--decode-mode",    default="beam",
                        choices=["greedy", "beam", "aed"],
                        help="Modo de decodificación para evaluación")
    args = parser.parse_args()

    # ── Verificar dependencias ─────────────────────────────────────────────
    _check_deps()

    # ── Verificar datos locales ────────────────────────────────────────────
    _verify_local_data()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device: {device}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fase 1: Frontend de audio ──────────────────────────────────────────
    log.info("\n[FASE 1] Inicializando frontend de audio...")
    audio_frontend = AudioFrontend(
        sample_rate=16_000,
        n_mels=80,
        n_fft=512,
        win_length=400,
        hop_length=160,
        pre_emphasis=0.97,
        cmvn=True,
    )

    # ── Fase 3: Tokenizador ────────────────────────────────────────────────
    log.info("\n[FASE 3] Preparando tokenizador BPE español peninsular...")
    tokenizer = SpanishBPETokenizer(
        vocab_size=args.vocab_size,
        model_dir=str(output_dir / "tokenizer"),
    )

    # ── Fase 6: Datasets ───────────────────────────────────────────────────
    log.info("\n[FASE 6] Cargando datasets desde rutas locales...")
    train_loader, val_loader, corpus_texts = build_dataloaders(
        tokenizer=tokenizer,
        audio_frontend=audio_frontend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Entrenar tokenizer si no existe
    if not tokenizer.is_trained:
        log.info("\n[FASE 3b] Entrenando tokenizer BPE...")
        tokenizer.train(corpus_texts)
    else:
        log.info("Cargando tokenizer existente...")
        if not tokenizer.sp:
            tokenizer.load()

    vocab_size = tokenizer.vocab_size_real()
    log.info(f"Vocabulario real: {vocab_size} tokens")

    # ── Validar tokenizer ──────────────────────────────────────────────────
    if tokenizer.sp is None:
        log.error("El tokenizer no se inicializó correctamente. Verifica los textos del corpus.")
        sys.exit(1)
    try:
        _test_ids = tokenizer.encode("hola mundo", add_sos=True, add_eos=True)
        log.info(f"✓ Tokenizer OK — 'hola mundo' → {len(_test_ids)} tokens")
    except Exception as _tok_exc:
        log.error(f"El tokenizer no puede codificar texto: {_tok_exc}")
        sys.exit(1)

    # ── Fase 2 + 4: Modelo ASSR ───────────────────────────────────────────
    log.info("\n[FASE 2+4] Construyendo modelo ASSR Conformer...")
    model = ASSR(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        conv_kernel_size=31,
        ff_expansion=4,
        dropout=args.dropout,
        ctc_weight=0.3,
        n_mels=80,
    )
    log.info(f"Parámetros entrenables: {model.num_parameters():,}")

    # Guardar config del modelo
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "num_heads": args.num_heads,
        "conv_kernel_size": 31,
        "ff_expansion": 4,
        "dropout": args.dropout,
        "ctc_weight": 0.3,
        "n_mels": 80,
    }
    (output_dir / "model_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Evaluación standalone ──────────────────────────────────────────────
    if args.eval_only:
        ckpt_path = args.resume or str(output_dir / "checkpoints" / "best.pt")
        if not pathlib.Path(ckpt_path).exists():
            log.error(f"No se encontró checkpoint en {ckpt_path}")
            sys.exit(1)
        log.info(f"Cargando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        log.info("\n[FASE 8] Evaluación...")
        metrics = evaluate(
            model, val_loader, tokenizer, device,
            decode_mode=args.decode_mode,
        )
        results_path = output_dir / "eval_results.json"
        results_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Resultados guardados en {results_path}")
        return

    # ── Fase 5: Entrenamiento ─────────────────────────────────────────────
    # Validar features de audio con el primer batch antes de entrenar
    log.info("\n[FASE 5] Validando features de audio...")
    try:
        _feat_batch, _tgt_batch, _feat_len, _tgt_len = next(iter(train_loader))
        log.info(
            f"✓ Audio features — shape={tuple(_feat_batch.shape)}, "
            f"min={_feat_batch.min():.3f}, max={_feat_batch.max():.3f}, "
            f"mean={_feat_batch.mean():.3f}"
        )
        if torch.isnan(_feat_batch).any() or torch.isinf(_feat_batch).any():
            log.error("Se detectaron NaN/Inf en las features de audio. "
                      "Verifica VAD y normalización.")
            sys.exit(1)
        del _feat_batch, _tgt_batch, _feat_len, _tgt_len
    except StopIteration:
        log.error("El DataLoader de entrenamiento está vacío. Verifica la carga de datos.")
        sys.exit(1)

    log.info("\n[FASE 5] Iniciando entrenamiento...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        resume_from=args.resume,
    )

    # ── Fase 7 + 8: Decodificación y Evaluación ───────────────────────────
    log.info("\n[FASE 7+8] Decodificación y evaluación final...")
    best_ckpt = output_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info("Mejor checkpoint cargado para evaluación final.")

    # Greedy (rápido, baseline)
    log.info("-- Greedy CTC --")
    metrics_greedy = evaluate(
        model, val_loader, tokenizer, device, decode_mode="greedy", max_batches=10
    )

    # Beam search (calidad)
    log.info("-- Beam Search (beam=8) --")
    metrics_beam = evaluate(
        model, val_loader, tokenizer, device, decode_mode="beam",
        beam_size=8, max_batches=10,
    )

    # Guardar resultados
    results = {
        "greedy": metrics_greedy,
        "beam_search": metrics_beam,
    }
    results_path = output_dir / "eval_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"\nResultados de evaluación guardados en {results_path}")
    log.info(
        f"\n{'='*60}\n"
        f"  WER (greedy):      {metrics_greedy['wer']*100:.2f}%\n"
        f"  WER (beam=8):      {metrics_beam['wer']*100:.2f}%\n"
        f"  CER (greedy):      {metrics_greedy['cer']*100:.2f}%\n"
        f"  CER (beam=8):      {metrics_beam['cer']*100:.2f}%\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
