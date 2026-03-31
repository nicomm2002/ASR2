# ASSR Español de España

Sistema completo de **Reconocimiento Automático de Voz (ASR)** para el **español peninsular (castellano de España)**, construido desde cero en PyTorch.

---

## Arquitectura

| Componente | Descripción |
|---|---|
| **Frontend** | Mel filterbank 80 filtros, 25 ms / 10 ms, preénfasis 0.97, VAD peninsular, CMVN |
| **Encoder** | Conformer 17 bloques, d_model=512, RoPE, subsampling CNN 8× |
| **Tokenizador** | BPE SentencePiece, vocab 6000, español peninsular (ñ, acentos, ll/ch) |
| **Decoder CTC** | Decoder principal (velocidad), peso loss = 0.3 |
| **Decoder AED** | Transformer 4 capas (precisión), peso loss = 0.7 |
| **Loss híbrida** | 0.3 · L_CTC + 0.7 · L_CE |
| **Decodificación** | Greedy CTC + Beam Search beam=8 + AED autoregresivo |

Basado en:
- [Whisper (OpenAI, 2022)](https://arxiv.org/abs/2212.04356)
- [Conformer (Google, 2020)](https://arxiv.org/abs/2005.08100)
- [FastConformer (NVIDIA, 2023)](https://arxiv.org/abs/2305.05084)
- [wav2vec 2.0 (Meta, 2020)](https://arxiv.org/abs/2006.11477)
- [Omnilingual ASR (Meta, 2025)](https://arxiv.org/abs/2511.09690)
- [Samba-ASR (2025)](https://arxiv.org/abs/2501.02356)

---

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

> **GPU recomendada.** Para CPU es funcional pero lento en entrenamiento.  
> Python ≥ 3.9 requerido.

---

## Uso — Pipeline completo con un solo comando

```bash
python asr_espanol_espana.py
```

El script ejecuta automáticamente **todas las fases**:

1. **Descarga** Common Voice 16.0 ES (filtrado España) y corpus RTVE 2022
2. **Filtra** muestras peninsulares (excluye México, Argentina, Colombia, etc.)
3. **Entrena** el tokenizador BPE (6 000 tokens, español peninsular)
4. **Construye** los DataLoaders con speed perturbation y SpecAugment
5. **Entrena** el modelo Conformer durante N épocas con checkpointing
6. **Evalúa** con WER/CER + análisis específico de rasgos peninsulares

### Opciones de línea de comandos

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--data-dir` | `data` | Directorio para datasets |
| `--output-dir` | `output` | Checkpoints, tokenizer, resultados |
| `--epochs` | `50` | Número de épocas de entrenamiento |
| `--batch-size` | `8` | Tamaño de batch |
| `--lr` | `5e-4` | Learning rate inicial |
| `--warmup-steps` | `10000` | Pasos de warmup lineal |
| `--d-model` | `512` | Dimensión del modelo |
| `--encoder-layers` | `17` | Bloques Conformer |
| `--decoder-layers` | `4` | Capas AED |
| `--vocab-size` | `6000` | Tamaño vocabulario BPE |
| `--device` | auto | `cuda` o `cpu` |
| `--resume` | — | Ruta a checkpoint para reanudar |
| `--eval-only` | — | Solo evaluar (requiere `--resume`) |
| `--decode-mode` | `beam` | `greedy` \| `beam` \| `aed` |

### Ejemplos

```bash
# Entrenamiento en GPU con batch más grande
python asr_espanol_espana.py --epochs 100 --batch-size 16 --device cuda

# Reanudar entrenamiento desde checkpoint
python asr_espanol_espana.py --resume output/checkpoints/last.pt

# Solo evaluar un modelo entrenado
python asr_espanol_espana.py --eval-only --resume output/checkpoints/best.pt --decode-mode beam
```

---

## Datasets

| Dataset | Fuente | Filtro |
|---|---|---|
| **Common Voice 16.0 ES** | HuggingFace `mozilla-foundation/common_voice_16_0` | Solo acento España |
| **RTVE 2022** | HuggingFace `projecte-aina/rtve2022asr` / Zenodo | TV y radio española |

**Excluidos explícitamente:** México, Argentina, Colombia, Chile, Perú, Venezuela, Ecuador, Bolivia, Paraguay, Uruguay, Cuba, República Dominicana, Puerto Rico, Costa Rica, Guatemala, Honduras, Nicaragua, El Salvador, Panamá.

---

## Métricas de evaluación

- **WER** (Word Error Rate) — solo hablantes peninsulares
- **CER** (Character Error Rate) — solo hablantes peninsulares
- **Distinción c/z** — porcentaje de errores en pares mínimos peninsulares (caza/casa, cena/sena)
- **Recuperación de vosotros** — recall de formas verbales vosotros (-áis, -éis)

Los resultados se guardan en `output/eval_results.json`.

---

## Estructura del proyecto

```
ASR2/
├── asr_espanol_espana.py   # Pipeline completo (fases 1-8)
├── requirements.txt         # Dependencias Python
├── README.md
├── data/                    # Datasets (generado automáticamente)
│   ├── common_voice/
│   └── rtve2022/
└── output/                  # Salidas (generado automáticamente)
    ├── tokenizer/
    │   ├── es_peninsular.model
    │   └── es_peninsular.vocab
    ├── checkpoints/
    │   ├── best.pt
    │   └── last.pt
    ├── model_config.json
    └── eval_results.json
```

---

## Fases del pipeline

| Fase | Descripción |
|---|---|
| 1 | Frontend de audio: 16 kHz mono, VAD peninsular, preénfasis, Mel 80, log, CMVN |
| 2 | Encoder Conformer: subsampling 8×, proyección 80→512, 17 bloques RoPE, LayerNorm |
| 3 | Tokenización BPE: corpus peninsular, vocab 6000, tokens especiales, conserva ñ/acentos |
| 4 | Decoder CTC + AED Transformer 4 capas, loss híbrida 0.3·CTC + 0.7·AED |
| 5 | Entrenamiento: SpecAugment (español), speed perturbation, AdamW + warmup + cosine |
| 6 | Datasets: Common Voice ES + RTVE 2022, filtro peninsular, exclusión latinoamericana |
| 7 | Decodificación: greedy, beam search beam=8, AED autoregresivo |
| 8 | Evaluación: WER/CER hablantes peninsulares, análisis c/z, vosotros, léxico |
