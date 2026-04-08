# mt_llmkit — Example App

Demonstrates how to use the [mt_llmkit](https://github.com/mobitouchOS/mt_llmkit) Flutter plugin for running Large Language Models locally on device via llama.cpp.

## Features demonstrated

- **Local GGUF** — download and run a Llama model on-device with real-time streaming output and performance metrics
- **REST API** — send prompts to remote AI providers
- **Vision** — multimodal inference with image input
- **RAG** — Retrieval-Augmented Generation with local document indexing

## Getting started

```bash
flutter pub get
flutter run
```

On first launch, tap **Download GGUF model** to fetch `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (~800 MB) from HuggingFace. The model is stored in the app's documents directory and reused on subsequent launches.

## Running tests

```bash
# Widget and unit tests
flutter test

# Integration tests (requires a connected device)
flutter test integration_test/
```
