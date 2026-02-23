// lib/src/rag/embeddings/llama_embedding_provider.dart

import 'dart:io';
import 'dart:isolate';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../core/llm_config.dart';
import '../../native/library_loader.dart';
import 'embedding_provider.dart';

// ── Worker Isolate entry point ─────────────────────────────────────────────

/// Wejście do worker isolate embeddingów.
///
/// ## Dlaczego własny Isolate zamiast LlamaParent?
///
/// `LlamaParent.getEmbeddings()` ulega crashowi z błędem:
/// "Cannot invoke native callback from a different isolate."
///
/// Przyczyna: `llama_decode` (wywoływane przez `Llama.getEmbeddings()`)
/// trigguje natywny log callback zarejestrowany jako `NativeCallable.isolateLocal`.
/// Jeśli ten callback był zarejestrowany w głównym Isolate, a wywołanie
/// pochodzi z child Isolate (jak w `LlamaChild`), Dart VM rzuca fatal error.
///
/// Rozwiązanie: zbudować własny Isolate, w którym `LibraryLoader.initialize()`
/// i konstruktor `Llama()` są wywołane **wewnątrz** tego Isolate.
/// Dzięki temu `NativeCallable.isolateLocal` rejestruje callback powiązany
/// z tym właśnie Isolate — wywołania FFI z tego Isolate są bezpieczne.
void _embeddingWorkerMain(Map<String, dynamic> args) {
  final String modelPath = args['modelPath'] as String;
  final int nGpuLayers = args['nGpuLayers'] as int;
  final int nCtx = args['nCtx'] as int;
  final int nBatch = args['nBatch'] as int;
  final int nThreads = args['nThreads'] as int;
  final SendPort mainPort = args['sendPort'] as SendPort;

  // KLUCZOWE: wywołaj LibraryLoader.initialize() TUTAJ, w tym Isolate.
  // Rejestruje to natywne callbacki (log, decode) powiązane z tym Isolate.
  LibraryLoader.initialize();

  Llama? llama;
  try {
    llama = Llama(
      modelPath,
      modelParams: ModelParams()
        ..nGpuLayers = nGpuLayers
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..embeddings = true               // tryb embeddingów (wyłącza generowanie)
        ..poolingType = LlamaPoolingType.mean  // mean pooling — standard dla RAG
        ..nCtx = nCtx
        ..nBatch = nBatch
        ..nThreads = nThreads,
      samplerParams: SamplerParams(),     // irrelevant w trybie embeddings
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'message': '$e'});
    return;
  }

  // Poinformuj główny Isolate że model jest gotowy — wyślij swój SendPort
  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  // Pętla obsługi żądań — worker żyje przez cały czas lifecycle providera
  receivePort.listen((dynamic message) {
    if (message is! Map<String, dynamic>) return;

    switch (message['type'] as String?) {
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          // getEmbeddings() działa bezpiecznie tutaj — callback zarejestrowany
          // w TYM Isolate, więc nie ma konfliktu isolate/callback
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

/// Implementacja [EmbeddingProvider] używająca lokalnego modelu GGUF
/// przez llama.cpp w dedykowanym Dart Isolate.
///
/// ## Architektura — własny Isolate (nie LlamaParent)
///
/// ```
/// embed(text)
///    │  SendPort.send({'type': 'embed', 'text': text, 'replyPort': ...})
///    ▼
/// Worker Isolate
///    │  LibraryLoader.initialize()  ← log callbacks w tym Isolate
///    │  Llama(path, embeddings=true)
///    │  llama.getEmbeddings(text)   ← bezpieczne FFI
///    │  SendPort.send({'type': 'ok', 'embedding': [...]})
///    ▼
/// List<double>                      ← wynik w głównym Isolate
/// ```
///
/// ## Wymagany model
///
/// Dedykowany model embeddingów w formacie GGUF:
/// - `nomic-embed-text-v1.5.Q4_K_M.gguf` (~270MB) — rekomendowany
/// - `bge-small-en-v1.5.Q4_K_M.gguf` (~67MB) — mniejszy, nieco słabszy
///
/// ## Użycie
///
/// ```dart
/// final provider = LlamaEmbeddingProvider();
/// await provider.initialize({'modelPath': '/path/to/nomic-embed.gguf'});
///
/// final vec = await provider.embed('Jaka jest stolica Polski?');
/// print(vec.length);  // 768
///
/// await provider.dispose();
/// ```
class LlamaEmbeddingProvider implements EmbeddingProvider {
  Isolate? _isolate;
  SendPort? _workerPort;
  int _dimensions = 0;
  bool _isInitialized = false;

  /// Inicjalizuje provider — spawuje worker Isolate i ładuje model.
  ///
  /// Wymagane klucze w [config]:
  ///   - `modelPath` (String): ścieżka do pliku .gguf modelu embeddingów
  ///
  /// Opcjonalne klucze:
  ///   - `llmConfig` (LlmConfig): nadpisuje domyślne parametry
  ///   - `nCtx` (int): rozmiar kontekstu (domyślnie 512 — wystarczy dla embeddingów)
  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final modelPath = config['modelPath'] as String?;
    if (modelPath == null || modelPath.isEmpty) {
      throw ArgumentError('Klucz "modelPath" jest wymagany dla embedding provider');
    }
    if (!File(modelPath).existsSync()) {
      throw FileSystemException('Plik modelu embeddingów nie istnieje', modelPath);
    }

    final llmConfig = config['llmConfig'] as LlmConfig? ?? const LlmConfig();

    // Port do odbioru odpowiedzi inicjalizacyjnej
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

    // Czekamy na 'ready' lub 'error' z worker Isolate
    final initResponse = await initReceivePort.first as Map<String, dynamic>;
    initReceivePort.close();

    if (initResponse['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw Exception(
        'Błąd ładowania modelu embeddingów: ${initResponse['message']}',
      );
    }

    _workerPort = initResponse['port'] as SendPort;

    // Ustal wymiarowość wektora przez testowy embed
    final testVec = await embed('dimension_probe');
    _dimensions = testVec.length;
    _isInitialized = true;
  }

  /// Generuje wektor embeddingu dla [text].
  ///
  /// Wywołanie jest nieblokujące — kompute odbywa się w worker Isolate.
  @override
  Future<List<double>> embed(String text) async {
    if (_workerPort == null) {
      throw StateError(
        'EmbeddingProvider nie jest zainicjalizowany. Wywołaj initialize() najpierw.',
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
      throw Exception('Błąd generowania embeddingu: ${response['message']}');
    }

    // embedding może być Float64List lub List<double> — normalizujemy
    final raw = response['embedding'] as List;
    return raw.map((e) => (e as num).toDouble()).toList();
  }

  /// Generuje embeddingi dla wielu tekstów sekwencyjnie.
  ///
  /// Każde wywołanie jest nieblokujące — UI może aktualizować się między nimi.
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

  // ── Prywatne helpery ─────────────────────────────────────────────────────

  /// Obcina tekst do ~2000 znaków (≈ 512 tokenów — limit kontekstu embeddings).
  String _truncateText(String text, {int maxChars = 2000}) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(' ');
    return lastSpace > maxChars * 0.8
        ? truncated.substring(0, lastSpace)
        : truncated;
  }
}
