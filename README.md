# Fine-Tuning Whisper for Kikuyu language

Fine-tuning [OpenAI's Whisper (small)](https://huggingface.co/openai/whisper-small) for automatic speech recognition (ASR) in **Kikuyu (Gĩkũyũ)**, a low-resource Bantu language spoken in Kenya. This repository contains the training notebook, inference examples, and a Gradio demo interface used to fine-tune and evaluate the model.

The resulting model is published at [NjeriKahoro/Kikuyu-Whisper](https://huggingface.co/NjeriKahoro/Kikuyu-Whisper).

## Overview

- **Base model:** `openai/whisper-small`
- **Task:** Automatic Speech Recognition (transcribe)
- **Language:** Kikuyu (`ki`)
- **Training dataset:** [`Anv-ke/kikuyu`](https://huggingface.co/datasets/Anv-ke/kikuyu) (streamed, `Audio(decode=False)`)
- **Framework:** 🤗 Transformers `Seq2SeqTrainer`
- **Environment:** Google Colab (GPU runtime), Google Drive for checkpoint storage

## Repository Contents

| File | Description |
|------|--------------|
| `Whisper.ipynb` | End-to-end notebook: dependency install, dataset loading & preprocessing, model fine-tuning, inference, and Gradio demo |

## Setup

```bash
pip install "datasets<4.0.0" transformers accelerate librosa soundfile
pip install -q evaluate jiwer librosa soundfile --upgrade
pip install rfc3987 jiwer gradio --upgrade
```

## Pipeline

### 1. Data Loading

The Kikuyu dataset is streamed directly from the Hugging Face Hub to avoid loading the full corpus into memory, with audio decoding deferred to handle malformed samples gracefully:

```python
dataset = load_dataset("Anv-ke/kikuyu", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False))
```

### 2. Preprocessing

Audio is decoded with `librosa` (more tolerant of irregular file headers than the default decoder), converted to log-Mel spectrogram input features, and transcriptions are tokenized into label IDs. Samples that fail to decode are caught and filtered out rather than crashing the pipeline:

A **seeded 50-example holdout** is reserved for evaluation, with the remainder used for training:

```python
train_dataset = dataset_processed["train"].skip(50)
eval_dataset = dataset_processed["train"].take(50)
```

### 3. Data Collation

A custom `DataCollatorSpeechSeq2SeqWithPadding` handles padding of variable-length audio features and label sequences, and includes a fallback that injects a silent dummy sample if an entire batch is corrupted — preventing GPU-side crashes during streaming.


> **Note:** Inference currently uses `language="swahili"` as a proxy language tag, since Whisper does not have a native Kikuyu language token. This is a known workaround — see [Limitations](#limitations) below.

## Interactive Demo

The notebook includes a Gradio interface for live transcription and WER scoring against a reference transcript:


## Limitations

- Whisper has no native Kikuyu language token, so `swahili` is used as the closest available proxy during decoding — this may introduce systematic biases in transcription.
- The dataset is streamed rather than fully downloaded, so some corrupted or malformed audio samples are silently skipped during preprocessing.

## Related

- Model card: [NjeriKahoro/Kikuyu-Whisper](https://huggingface.co/NjeriKahoro/Kikuyu-Whisper)
- Demo Space: [Omnilingual-Translation-Speech-RecognitionStudio](https://huggingface.co/spaces/NjeriKahoro/Omnilingual-Translation-Speech-RecognitionStudio)
- Dataset: [Anv-ke/kikuyu](https://huggingface.co/datasets/Anv-ke/kikuyu)

