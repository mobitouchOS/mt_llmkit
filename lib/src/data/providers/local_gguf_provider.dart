// lib/src/data/providers/local_gguf_provider.dart

import 'dart:async';
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../core/llm_config.dart';
import '../../domain/providers/llm_provider.dart';
import '../../native/library_loader.dart';

/// Implementation of [LLMProvider] for local GGUF models.
///
/// ## Data flow architecture
///
/// ```
/// UI Thread
///    │  listen()
///    ▼
/// StreamController<String>   ← safe thread boundary
///    ▲
///    │  add(token)
/// LlamaParent.stream         ← Dart Isolate (llama.cpp FFI)
///    │
///    ▼
/// llama.cpp native library   ← CPU/GPU computation
/// ```
///
/// [LlamaParent] runs generation in a separate Isolate, guaranteeing
/// no UI thread blocking. [StreamController.broadcast()] safely
/// passes tokens from the Isolate to consumers on the UI thread.
///
/// ## Usage
///
/// ```dart
/// final provider = LocalGGUFProvider();
/// await provider.initialize({
///   'modelPath': '/path/to/model.gguf',
///   'llmConfig': LlmConfig(temp: 0.7, nCtx: 2048),
/// });
///
/// await for (final token in provider.sendPrompt('Hello')) {
///   print(token); // tokens appear as they are generated
/// }
///
/// await provider.dispose();
/// ```
class LocalGGUFProvider implements LLMProvider {
  LlamaParent? _llamaParent;
  LlmConfig _config = const LlmConfig();
  bool _isInitialized = false;

  /// Initializes the provider — loads the GGUF model into an Isolate.
  ///
  /// Required keys in [config]:
  ///   - `modelPath` (String): path to the .gguf file
  ///
  /// Optional keys:
  ///   - `llmConfig` (LlmConfig): model parameters
  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final modelPath = config['modelPath'] as String?;
    if (modelPath == null || modelPath.isEmpty) {
      throw ArgumentError('Key "modelPath" is required in configuration');
    }
    if (!File(modelPath).existsSync()) {
      throw FileSystemException('Model file does not exist', modelPath);
    }

    // Initialize FFI native libraries for the current platform
    LibraryLoader.initialize();

    _config = config['llmConfig'] as LlmConfig? ?? const LlmConfig();

    final loadCommand = LlamaLoad(
      path: modelPath,
      verbose: false,
      modelParams: ModelParams()
        ..nGpuLayers = _config.nGpuLayersDefault
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..nCtx = _config.nCtxDefault
        ..nBatch = _config.nBatchDefault
        ..nPredict = _config.nPredictDefault
        ..nThreads = _config.nThreadsDefault,
      samplingParams: SamplerParams()
        ..temp = _config.tempDefault
        ..topK = _config.topKDefault
        ..topP = _config.topPDefault
        ..penaltyRepeat = _config.penaltyRepeatDefault,
    );

    // LlamaParent runs llama.cpp in a separate Isolate —
    // heavy computation does not block the UI thread
    _llamaParent = LlamaParent(loadCommand);
    await _llamaParent!.init();
    _isInitialized = true;
  }

  /// Sends a prompt to the model and returns a stream of tokens.
  ///
  /// Generation runs in an Isolate. Tokens are passed to the UI
  /// via [StreamController] — each `add()` is UI-thread-safe.
  ///
  /// Optional keys in [parameters]:
  ///   - `promptFormat` (PromptFormat): prompt format
  @override
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    if (!_isInitialized || _llamaParent == null) {
      throw StateError(
        'Provider is not initialized. Call initialize() first.',
      );
    }

    // StreamController.broadcast() — multiple listeners can subscribe
    // and it is safe to use as a bridge between the Isolate and the UI thread
    final controller = StreamController<String>.broadcast();

    final promptFormat =
        parameters?['promptFormat'] as PromptFormat? ??
        _config.promptFormatDefault;

    final formattedPrompt = promptFormat.formatPrompt(prompt);

    // Send the prompt to the Isolate — generation is asynchronous
    _llamaParent!.sendPrompt(formattedPrompt);

    // Forward each token from the Isolate stream to our StreamController.
    // This way the UI subscribes to one well-managed stream
    // instead of working directly with the Isolate stream.
    _llamaParent!.stream.listen(
      (token) {
        if (!controller.isClosed) {
          controller.add(token);
        }
      },
      onError: (Object error, StackTrace stackTrace) {
        if (!controller.isClosed) {
          controller.addError(error, stackTrace);
        }
      },
      onDone: () {
        if (!controller.isClosed) {
          controller.close();
        }
      },
      cancelOnError: false,
    );

    return controller.stream;
  }

  @override
  Future<void> dispose() async {
    _llamaParent?.dispose();
    _llamaParent = null;
    _isInitialized = false;
  }

  /// Whether the provider is initialized and ready for generation
  bool get isInitialized => _isInitialized;
}
