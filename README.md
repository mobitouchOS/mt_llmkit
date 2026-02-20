# llmcpp

A Flutter plugin for running Large Language Models (LLM) locally using llama.cpp.

## Features

✅ **Local LLM Inference** - Run language models directly on device  
✅ **Multiple Model Types** - Support for LlmModelIsolated and LlmModelStandard  
✅ **Streaming Generation** - Real-time text generation with streaming support  
✅ **Performance Metrics** - Built-in tokens/second (t/s) measurement  
✅ **Flexible Configuration** - Customize GPU layers, context size, temperature, and more  
✅ **Multiple Prompt Formats** - ChatML, Alpaca, Gemma, and custom formats  
✅ **Cross-Platform** - Works on Android and iOS

## Performance Metrics 🚀 NEW

Measure text generation performance with **live streaming metrics**:

```dart
final model = LlmModelIsolated(LlmConfig());
await model.loadModel('model.gguf');

// Streaming with LIVE performance metrics
model.sendPromptStream('Hello').listen((chunk) {
  print(chunk.text);  // Text in real-time
  
  if (chunk.metrics != null) {
    print('Speed: ${chunk.metrics!.tokensPerSecond.toStringAsFixed(2)} t/s');
    print('Tokens: ${chunk.metrics!.tokensGenerated}');
  }
});
```

**Three ways to generate:**
1. **sendPromptStream()** - ⭐ Best: Streaming + live metrics
2. **sendPrompt()** - Streaming only (no metrics)
3. **sendPromptWithMetrics()** - Wait for full response + metrics

See [PERFORMANCE_METRICS.md](PERFORMANCE_METRICS.md) for detailed documentation.

## Quick Start

### Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  llmcpp: ^1.0.0
```

### Basic Usage

```dart
import 'package:llmcpp/llmcpp.dart';

// Create model
final model = LlmModelIsolated(
  LlmConfig(
    temp: 0.7,
    nGpuLayers: 4,
    nCtx: 2048,
  ),
);

// Load model
await model.loadModel('/path/to/model.gguf');

// Streaming generation
model.sendPrompt('What is Flutter?')?.listen((chunk) {
  print(chunk);
});

// Or with performance metrics
final result = await model.sendPromptWithMetrics('What is Flutter?');
print(result.text);
print('Speed: ${result.metrics!.tokensPerSecond} t/s');

// Cleanup
model.dispose();
```

## Documentation

- [Performance Metrics Guide](PERFORMANCE_METRICS.md) - Tokens/second measurement
- [Example App](example/) - Full example application
- [API Reference](https://pub.dev/documentation/llmcpp/latest/)

## Example App

Check out the [example](example/) directory for a complete working application that demonstrates:
- Model downloading
- Text generation
- Performance monitoring
- UI integration

## Getting Started

This project is a starting point for a Flutter
[plug-in package](https://flutter.dev/to/develop-plugins),
a specialized package that includes platform-specific implementation code for
Android and/or iOS.

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

