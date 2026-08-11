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

The notebook is designed for Google Colab and mounts Google Drive for checkpoint/model persistence:

```python
from google.colab import drive
drive.mount('/content/drive')
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

```python
def prepare_dataset(batch):
    try:
        audio_bytes = batch["audio"]["bytes"]
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        batch["input_features"] = processor.feature_extractor(y, sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch
    except Exception as e:
        print(f"Bypassed broken sample: {e}")
        return {}
```

A **seeded 50-example holdout** is reserved for evaluation, with the remainder used for training:

```python
train_dataset = dataset_processed["train"].skip(50)
eval_dataset = dataset_processed["train"].take(50)
```

### 3. Data Collation

A custom `DataCollatorSpeechSeq2SeqWithPadding` handles padding of variable-length audio features and label sequences, and includes a fallback that injects a silent dummy sample if an entire batch is corrupted — preventing GPU-side crashes during streaming.

### 4. Training Configuration

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/whisper_kikuyu_final",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    warmup_steps=50,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    # ... (additional arguments — see notebook for full config)
)
```

> Effective batch size: 16 (1 × 16 gradient accumulation steps)

### 5. Evaluation

Word Error Rate (WER) is computed via the `evaluate` library during training:

```python
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}
```

### 6. Saving the Model

```python
trainer.save_model(final_path)
processor.save_pretrained(final_path)
```

## Inference

Standalone transcription on a saved audio file:

```python
model_path = "/content/drive/MyDrive/whisper_kikuyu_FINAL"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cuda")

speech, _ = librosa.load(audio_file, sr=16000)
input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

predicted_ids = model.generate(
    input_features,
    forced_decoder_ids=processor.get_decoder_prompt_ids(language="swahili", task="transcribe")
)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

> **Note:** Inference currently uses `language="swahili"` as a proxy language tag, since Whisper does not have a native Kikuyu language token. This is a known workaround — see [Limitations](#limitations) below.

## Interactive Demo

The notebook includes a Gradio interface for live transcription and WER scoring against a reference transcript:

```python
demo = gr.Interface(
    fn=transcribe_and_evaluate,
    inputs=[gr.Audio(type="filepath", label="Input Audio"), gr.Textbox(label="Reference Text")],
    outputs=[gr.Textbox(label="Prediction", lines=10), gr.Label(label="WER")],
    title="OpenAi Whisper Model Small",
)
demo.launch(share=True, debug=True)
```

## Limitations

- Whisper has no native Kikuyu language token, so `swahili` is used as the closest available proxy during decoding — this may introduce systematic biases in transcription.
- Training uses a small effective batch size (16) and a single held-out 50-sample evaluation set; results should be interpreted as preliminary rather than a robust benchmark.
- The dataset is streamed rather than fully downloaded, so some corrupted or malformed audio samples are silently skipped during preprocessing.

## Related

- Model card: [NjeriKahoro/Kikuyu-Whisper](https://huggingface.co/NjeriKahoro/Kikuyu-Whisper)
- Demo Space: [Omnilingual-Translation-Speech-RecognitionStudio](https://huggingface.co/spaces/NjeriKahoro/Omnilingual-Translation-Speech-RecognitionStudio)
- Dataset: [Anv-ke/kikuyu](https://huggingface.co/datasets/Anv-ke/kikuyu)

## License

Apache 2.0 (matching base Whisper license) — confirm/update as appropriate.
