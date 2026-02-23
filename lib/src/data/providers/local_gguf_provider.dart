// lib/src/data/providers/local_gguf_provider.dart

import 'dart:async';
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../core/llm_config.dart';
import '../../domain/providers/llm_provider.dart';
import '../../native/library_loader.dart';

/// Implementacja [LLMProvider] dla lokalnych modeli GGUF.
///
/// ## Architektura przepływu danych
///
/// ```
/// UI Thread
///    │  listen()
///    ▼
/// StreamController<String>   ← bezpieczna granica wątków
///    ▲
///    │  add(token)
/// LlamaParent.stream         ← Dart Isolate (llama.cpp FFI)
///    │
///    ▼
/// llama.cpp native library   ← obliczenia CPU/GPU
/// ```
///
/// [LlamaParent] uruchamia generowanie w osobnym Isolate, co gwarantuje
/// brak blokowania wątku UI. [StreamController.broadcast()] bezpiecznie
/// przekazuje tokeny z Isolate do konsumentów na UI thread.
///
/// ## Użycie
///
/// ```dart
/// final provider = LocalGGUFProvider();
/// await provider.initialize({
///   'modelPath': '/path/to/model.gguf',
///   'llmConfig': LlmConfig(temp: 0.7, nCtx: 2048),
/// });
///
/// await for (final token in provider.sendPrompt('Hello')) {
///   print(token); // tokenyi pojawiają się na bieżąco
/// }
///
/// await provider.dispose();
/// ```
class LocalGGUFProvider implements LLMProvider {
  LlamaParent? _llamaParent;
  LlmConfig _config = const LlmConfig();
  bool _isInitialized = false;

  /// Inicjalizuje provider — ładuje model GGUF do Isolate.
  ///
  /// Wymagane klucze w [config]:
  ///   - `modelPath` (String): ścieżka do pliku .gguf
  ///
  /// Opcjonalne klucze:
  ///   - `llmConfig` (LlmConfig): parametry modelu
  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final modelPath = config['modelPath'] as String?;
    if (modelPath == null || modelPath.isEmpty) {
      throw ArgumentError('Klucz "modelPath" jest wymagany w konfiguracji');
    }
    if (!File(modelPath).existsSync()) {
      throw FileSystemException('Plik modelu nie istnieje', modelPath);
    }

    // Inicjalizacja natywnych bibliotek FFI dla bieżącej platformy
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

    // LlamaParent uruchamia llama.cpp w osobnym Isolate —
    // ciężkie obliczenia nie blokują wątku UI
    _llamaParent = LlamaParent(loadCommand);
    await _llamaParent!.init();
    _isInitialized = true;
  }

  /// Wysyła prompt do modelu i zwraca stream tokenów.
  ///
  /// Generowanie odbywa się w Isolate. Tokeny są przekazywane do UI
  /// przez [StreamController] — każdy `add()` jest bezpieczny dla wątku UI.
  ///
  /// Opcjonalne klucze w [parameters]:
  ///   - `promptFormat` (PromptFormat): format promptu
  @override
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    if (!_isInitialized || _llamaParent == null) {
      throw StateError(
        'Provider nie jest zainicjalizowany. Wywołaj initialize() najpierw.',
      );
    }

    // StreamController.broadcast() — wiele listenerów może subskrybować
    // i jest bezpieczny do użycia jako most między Isolate a UI thread
    final controller = StreamController<String>.broadcast();

    final promptFormat =
        parameters?['promptFormat'] as PromptFormat? ??
        _config.promptFormatDefault;

    final formattedPrompt = promptFormat.formatPrompt(prompt);

    // Wysyłamy prompt do Isolate — generowanie jest asynchroniczne
    _llamaParent!.sendPrompt(formattedPrompt);

    // Przekazujemy każdy token z Isolate stream do naszego StreamController.
    // Dzięki temu UI subskrybuje jeden, dobrze zarządzany stream
    // zamiast bezpośrednio pracować ze streamem Isolate.
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

  /// Czy provider jest zainicjalizowany i gotowy do generowania
  bool get isInitialized => _isInitialized;
}
