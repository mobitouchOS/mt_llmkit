# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmcpp** is a Flutter plugin that enables running Large Language Models (LLMs) locally on Android and iOS using the `llama.cpp` library via FFI (`llamadart ^0.6.8`). It provides real-time streaming inference and performance metrics.

## Commands

```bash
# Install dependencies
flutter pub get

# Run all tests
flutter test

# Run a specific test file
flutter test test/src/core/performance_metrics_test.dart

# Run tests with coverage
flutter test --coverage

# Lint
flutter analyze

# Format
dart format lib/ test/ example/lib/

# Run the example app
cd example && flutter pub get && flutter run
```

## Architecture

### Public API

`lib/llmcpp.dart` is the single export file. It re-exports everything from `src/` plus `LlamaImageContent`, `LlamaTextContent`, `LlamaContentPart` from `llamadart`.

### Class Hierarchy

```
LlmInterface (abstract interface)
  └─ LlmModelBase (abstract, shared state management)
       ├─ LlmModelIsolated  → wraps LlamaEngine in a Dart Isolate
       └─ LlmModelStandard  → wraps LlamaEngine directly (in-process)
```

Both model classes share the same lifecycle: `loadModel(path)` → generate → `dispose()`.

### Three Generation Methods

1. `sendPrompt(prompt) → Stream<String>` — pure streaming, no metrics
2. `sendPromptWithMetrics(prompt) → Future<GenerationResult>` — full response with metrics (deprecated)
3. `sendPromptStream(prompt) → Stream<StreamingChunk>` — **recommended**: live streaming + real-time metrics

`StreamingChunk` carries: `text`, `metrics` (PerformanceMetrics), and `isFinal` flag.

### Configuration

`LlmConfig` is an immutable config object. Key parameters: `temp`, `nGpuLayers`, `nCtx`, `nBatch`, `nThreads`, `topK`, `topP`, `penaltyRepeat`, `mmprojPath`.

### Prompt Formats (Strategy pattern)

Set via `LlmConfig.promptFormat`. Built-in: `ChatMLFormat()`, `AlpacaFormat()`, `GemmaFormat()` (from `llama_cpp_dart`). Custom: `HarmonyFormat()` in `lib/src/formats/harmony_format.dart`.

### Performance Metrics

`PerformanceMetrics` tracks `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken`. Token counting is exact: each callback from llama.cpp emits one token, counted via `+= 1` in the streaming loop.

### Native Libraries

- Android: pre-compiled `.so` files in `android/src/main/jniLibs/{arm64-v8a,x86_64}/`
- iOS: native frameworks via CocoaPods (`ios/llmcpp.podspec`)
- Loading managed by `llamadart` via Dart Build Hooks (no manual preloading required)

## Test Infrastructure

Tests live in `test/src/{core,models,formats}/`. Shared helpers are in `test/helpers/test_helpers.dart`:
- `TestHelpers` — factory methods and matchers
- `TestConfigBuilder` — fluent builder for `LlmConfig` in tests
- Pre-built minimal/maximal configs for common test scenarios
