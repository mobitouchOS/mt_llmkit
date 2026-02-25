// lib/src/rag/embeddings/llama_embedding_provider.dart

import 'dart:io';
import 'dart:isolate';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../core/llm_config.dart';
import '../../native/library_loader.dart';
import 'embedding_provider.dart';

// ── Worker Isolate entry point ─────────────────────────────────────────────

/// Entry point for the embedding worker isolate.
///
/// ## Why a custom Isolate instead of LlamaParent?
///
/// `LlamaParent.getEmbeddings()` crashes with the error:
/// "Cannot invoke native callback from a different isolate."
///
/// Cause: `llama_decode` (called by `Llama.getEmbeddings()`)
/// triggers a native log callback registered as `NativeCallable.isolateLocal`.
/// If that callback was registered in the main Isolate, but the call
/// originates from a child Isolate (as in `LlamaChild`), the Dart VM throws a fatal error.
///
/// Solution: build a custom Isolate in which `LibraryLoader.initialize()`
/// and the `Llama()` constructor are called **inside** that Isolate.
/// This way `NativeCallable.isolateLocal` registers a callback bound
/// to that very Isolate — FFI calls from that Isolate are safe.
void _embeddingWorkerMain(Map<String, dynamic> args) {
  final String modelPath = args['modelPath'] as String;
  final int nGpuLayers = args['nGpuLayers'] as int;
  final int nCtx = args['nCtx'] as int;
  final int nBatch = args['nBatch'] as int;
  final int nThreads = args['nThreads'] as int;
  final SendPort mainPort = args['sendPort'] as SendPort;

  // CRITICAL: call LibraryLoader.initialize() HERE, inside this Isolate.
  // This registers native callbacks (log, decode) bound to this Isolate.
  LibraryLoader.initialize();

  Llama? llama;
  try {
    llama = Llama(
      modelPath,
      modelParams: ModelParams()
        ..nGpuLayers = nGpuLayers
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..embeddings = true               // embeddings mode (disables generation)
        ..poolingType = LlamaPoolingType.mean  // mean pooling — standard for RAG
        ..nCtx = nCtx
        ..nBatch = nBatch
        ..nThreads = nThreads,
      samplerParams: SamplerParams(),     // irrelevant in embeddings mode
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'message': '$e'});
    return;
  }

  // Notify the main Isolate that the model is ready — send our SendPort
  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  // Request handling loop — the worker lives for the entire provider lifecycle
  receivePort.listen((dynamic message) {
    if (message is! Map<String, dynamic>) return;

    switch (message['type'] as String?) {
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          // getEmbeddings() is safe here — callback registered
          // in THIS Isolate, so there is no isolate/callback conflict
          final embedding = llama!.getEmbeddings(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', 'message': '$e'});
        }

      case 'dispose':
        llama?.dispose();
        receivePort.close();
    }
  });
}

// ── LlamaEmbeddingProvider ─────────────────────────────────────────────────

/// Implementation of [EmbeddingProvider] using a local GGUF model
/// via llama.cpp in a dedicated Dart Isolate.
///
/// ## Architecture — custom Isolate (not LlamaParent)
///
/// ```
/// embed(text)
///    │  SendPort.send({'type': 'embed', 'text': text, 'replyPort': ...})
///    ▼
/// Worker Isolate
///    │  LibraryLoader.initialize()  ← log callbacks in this Isolate
///    │  Llama(path, embeddings=true)
///    │  llama.getEmbeddings(text)   ← safe FFI
///    │  SendPort.send({'type': 'ok', 'embedding': [...]})
///    ▼
/// List<double>                      ← result in the main Isolate
/// ```
///
/// ## Required model
///
/// A dedicated embeddings model in GGUF format:
/// - `nomic-embed-text-v1.5.Q4_K_M.gguf` (~270MB) — recommended
/// - `bge-small-en-v1.5.Q4_K_M.gguf` (~67MB) — smaller, slightly weaker
///
/// ## Usage
///
/// ```dart
/// final provider = LlamaEmbeddingProvider();
/// await provider.initialize({'modelPath': '/path/to/nomic-embed.gguf'});
///
/// final vec = await provider.embed('What is the capital of Poland?');
/// print(vec.length);  // 768
///
/// await provider.dispose();
/// ```
class LlamaEmbeddingProvider implements EmbeddingProvider {
  Isolate? _isolate;
  SendPort? _workerPort;
  int _dimensions = 0;
  bool _isInitialized = false;

  /// Initializes the provider — spawns the worker Isolate and loads the model.
  ///
  /// Required keys in [config]:
  ///   - `modelPath` (String): path to the .gguf embeddings model file
  ///
  /// Optional keys:
  ///   - `llmConfig` (LlmConfig): overrides default parameters
  ///   - `nCtx` (int): context size (default 512 — sufficient for embeddings)
  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final modelPath = config['modelPath'] as String?;
    if (modelPath == null || modelPath.isEmpty) {
      throw ArgumentError('Key "modelPath" is required for embedding provider');
    }
    if (!File(modelPath).existsSync()) {
      throw FileSystemException('Embedding model file does not exist', modelPath);
    }

    final llmConfig = config['llmConfig'] as LlmConfig? ?? const LlmConfig();

    // Port for receiving the initialization response
    final initReceivePort = ReceivePort();

    _isolate = await Isolate.spawn(
      _embeddingWorkerMain,
      {
        'modelPath': modelPath,
        'nGpuLayers': llmConfig.nGpuLayersDefault,
        'nCtx': (config['nCtx'] as int?) ?? 512,
        'nBatch': llmConfig.nBatchDefault,
        'nThreads': llmConfig.nThreadsDefault,
        'sendPort': initReceivePort.sendPort,
      },
      debugName: 'llmcpp_EmbeddingWorker',
    );

    // Wait for 'ready' or 'error' from the worker Isolate
    final initResponse = await initReceivePort.first as Map<String, dynamic>;
    initReceivePort.close();

    if (initResponse['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw Exception(
        'Failed to load embedding model: ${initResponse['message']}',
      );
    }

    _workerPort = initResponse['port'] as SendPort;

    // Determine vector dimensionality via a test embed
    final testVec = await embed('dimension_probe');
    _dimensions = testVec.length;
    _isInitialized = true;
  }

  /// Generates an embedding vector for [text].
  ///
  /// Non-blocking — computation runs in the worker Isolate.
  @override
  Future<List<double>> embed(String text) async {
    if (_workerPort == null) {
      throw StateError(
        'EmbeddingProvider is not initialized. Call initialize() first.',
      );
    }

    final replyPort = ReceivePort();
    _workerPort!.send({
      'type': 'embed',
      'text': _truncateText(text),
      'replyPort': replyPort.sendPort,
    });

    final response = await replyPort.first as Map<String, dynamic>;
    replyPort.close();

    if (response['type'] == 'error') {
      throw Exception('Embedding generation failed: ${response['message']}');
    }

    // embedding may be Float64List or List<double> — normalise
    final raw = response['embedding'] as List;
    return raw.map((e) => (e as num).toDouble()).toList();
  }

  /// Generates embeddings for multiple texts sequentially.
  ///
  /// Each call is non-blocking — the UI can update between them.
  @override
  Future<List<List<double>>> embedBatch(List<String> texts) async {
    final results = <List<double>>[];
    for (final text in texts) {
      results.add(await embed(text));
    }
    return results;
  }

  @override
  Future<void> dispose() async {
    _workerPort?.send({'type': 'dispose'});
    await Future.delayed(const Duration(milliseconds: 100)); // give worker time to cleanup
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _workerPort = null;
    _dimensions = 0;
    _isInitialized = false;
  }

  @override
  int get dimensions => _dimensions;

  @override
  bool get isInitialized => _isInitialized;

  // ── Private helpers ──────────────────────────────────────────────────────

  /// Truncates text to ~2000 chars (≈ 512 tokens — embeddings context limit).
  String _truncateText(String text, {int maxChars = 2000}) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(' ');
    return lastSpace > maxChars * 0.8
        ? truncated.substring(0, lastSpace)
        : truncated;
  }
}
