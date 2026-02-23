// lib/src/rag/llama_rag_coordinator.dart
//
// ARCHITEKTURA — dlaczego jeden izolat?
//
// llama.cpp przechowuje JEDEN globalny wskaźnik log callback (ustawiany przez
// `llama_log_set`). Dart tworzy izolato-lokalne `Pointer.fromFunction<>()`
// dla callbacków FFI — wywołanie takiego callbacku z innego izolatu powoduje
// FATAL CRASH: "Cannot invoke native callback from a different isolate."
//
// Gdy dwa izolaty (LlamaEmbeddingWorker + LlamaChild z LlamaParent) każdy
// wywołują LibraryLoader.initialize() → llama_log_set(own_isolate_callback),
// ostatni wygrywa wyścig. Wywołanie llama_decode przez "przegranego" niszczy
// aplikację, bo próbuje uruchomić callback z obcego izolatu.
//
// Rozwiązanie: JEDEN izolat zarządza oboma modelami. Jedno wywołanie
// LibraryLoader.initialize() → jedna rejestracja log callback → brak wyścigu.

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../core/llm_config.dart';
import '../domain/providers/llm_provider.dart';
import '../native/library_loader.dart';
import '../presentation/llm_plugin.dart';
import 'embeddings/embedding_provider.dart';

// ── Worker isolate ────────────────────────────────────────────────────────────

/// Wejście do worker isolate — zarządza modelem embeddingów ORAZ generowania.
///
/// Kluczowe: [LibraryLoader.initialize()] i oba konstruktory [Llama] są
/// wywoływane WEWNĄTRZ tego izolatu, więc `Pointer.fromFunction<>()` (log
/// callback) jest powiązany z tym właśnie izolatem. Brak konfliktu isolate.
void _llamaRagWorkerMain(Map<String, dynamic> args) {
  final String embedModelPath = args['embedModelPath'] as String;
  final String genModelPath = args['genModelPath'] as String;
  final int embedNCtx = args['embedNCtx'] as int;
  final int genNCtx = args['genNCtx'] as int;
  final int nGpuLayers = args['nGpuLayers'] as int;
  final int nBatch = args['nBatch'] as int;
  final int nThreads = args['nThreads'] as int;
  final double temp = (args['temp'] as num).toDouble();
  final int nPredict = args['nPredict'] as int;
  final int topK = args['topK'] as int;
  final double topP = (args['topP'] as num).toDouble();
  final double penaltyRepeat = (args['penaltyRepeat'] as num).toDouble();
  final SendPort mainPort = args['sendPort'] as SendPort;

  // KLUCZOWE: jedno wywołanie w tym izolate — jeden log callback.
  LibraryLoader.initialize();

  Llama? embedModel;
  Llama? genModel;

  try {
    embedModel = Llama(
      embedModelPath,
      modelParams: ModelParams()
        ..nGpuLayers = nGpuLayers
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..embeddings = true
        ..poolingType = LlamaPoolingType.mean
        ..nCtx = embedNCtx
        ..nBatch = nBatch
        ..nThreads = nThreads,
      samplerParams: SamplerParams(),
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'phase': 'embed_init', 'message': '$e'});
    return;
  }

  try {
    genModel = Llama(
      genModelPath,
      modelParams: ModelParams()
        ..nGpuLayers = nGpuLayers
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..nCtx = genNCtx
        ..nBatch = nBatch
        ..nThreads = nThreads
        ..nPredict = nPredict,
      samplerParams: SamplerParams()
        ..temp = temp
        ..topK = topK
        ..topP = topP
        ..penaltyRepeat = penaltyRepeat,
    );
  } catch (e) {
    embedModel.dispose();
    mainPort.send({'type': 'error', 'phase': 'gen_init', 'message': '$e'});
    return;
  }

  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  StreamSubscription<String>? genSubscription;

  receivePort.listen((dynamic message) {
    if (message is! Map<String, dynamic>) return;

    switch (message['type'] as String?) {
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          final embedding = embedModel!.getEmbeddings(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', 'message': '$e'});
        }

      case 'generate':
        final prompt = message['prompt'] as String;
        final streamPort = message['streamPort'] as SendPort;
        try {
          genModel!.setPrompt(prompt);
          genSubscription = genModel.generateText().listen(
            (token) => streamPort.send({'type': 'token', 'text': token}),
            onDone: () {
              streamPort.send({'type': 'done'});
              genSubscription = null;
            },
            onError: (Object e) {
              streamPort.send({'type': 'error', 'message': '$e'});
              genSubscription = null;
            },
          );
        } catch (e) {
          streamPort.send({'type': 'error', 'message': '$e'});
        }

      case 'stop_generate':
        genSubscription?.cancel();
        genSubscription = null;
        genModel?.clear();

      case 'dispose':
        genSubscription?.cancel();
        embedModel?.dispose();
        genModel?.dispose();
        receivePort.close();
    }
  });
}

// ── Prywatne implementacje providerów ─────────────────────────────────────────

class _CoordEmbeddingProvider implements EmbeddingProvider {
  final SendPort _workerPort;
  final int _dimensions;
  bool _isInitialized = true;

  _CoordEmbeddingProvider(this._workerPort, {required int dimensions})
      : _dimensions = dimensions;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    _isInitialized = true; // model już załadowany w koordynatorze
  }

  @override
  Future<List<double>> embed(String text) async {
    final replyPort = ReceivePort();
    _workerPort.send({
      'type': 'embed',
      'text': _truncate(text),
      'replyPort': replyPort.sendPort,
    });
    final response = await replyPort.first as Map<String, dynamic>;
    replyPort.close();
    if (response['type'] == 'error') {
      throw Exception('Błąd embeddingu: ${response['message']}');
    }
    final raw = response['embedding'] as List;
    return raw.map((e) => (e as num).toDouble()).toList();
  }

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
    _isInitialized = false;
  }

  @override
  int get dimensions => _dimensions;

  @override
  bool get isInitialized => _isInitialized;

  String _truncate(String text, {int maxChars = 2000}) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(' ');
    return lastSpace > maxChars * 0.8
        ? truncated.substring(0, lastSpace)
        : truncated;
  }
}

class _CoordGenerationProvider implements LLMProvider {
  final SendPort _workerPort;

  _CoordGenerationProvider(this._workerPort);

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    // model już załadowany w koordynatorze — no-op
  }

  @override
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    final controller = StreamController<String>();
    final replyPort = ReceivePort();

    _workerPort.send({
      'type': 'generate',
      'prompt': prompt,
      'streamPort': replyPort.sendPort,
    });

    final sub = replyPort.listen((dynamic message) {
      if (message is! Map<String, dynamic>) return;
      switch (message['type'] as String?) {
        case 'token':
          if (!controller.isClosed) {
            controller.add(message['text'] as String);
          }
        case 'done':
          replyPort.close();
          if (!controller.isClosed) controller.close();
        case 'error':
          replyPort.close();
          if (!controller.isClosed) {
            controller.addError(
              Exception('Błąd generowania: ${message['message']}'),
            );
          }
      }
    });

    controller.onCancel = () {
      sub.cancel();
      replyPort.close();
      _workerPort.send({'type': 'stop_generate'});
    };

    return controller.stream;
  }

  @override
  Future<void> dispose() async {
    // koordynator obsługuje dispose
  }
}

// ── LlamaRagCoordinator ───────────────────────────────────────────────────────

/// Koordynator zarządzający jednym worker isolate dla obu modeli llama.cpp.
///
/// ## Problem
///
/// `llama.cpp` przechowuje jeden globalny log callback. Dart tworzy
/// izolato-lokalne `Pointer.fromFunction<>()` dla callbacków FFI. Gdy dwa
/// izolaty (embedding + generowanie) równocześnie ustawiają `llama_log_set`,
/// wywołanie llama_decode w jednym z nich próbuje uruchomić callback z innego
/// izolatu → fatal crash: "Cannot invoke native callback from a different
/// isolate."
///
/// ## Rozwiązanie
///
/// Jeden izolat ([_llamaRagWorkerMain]) ładuje oba modele — model embeddingów
/// (z `embeddings=true`) i model generowania. Jedno wywołanie
/// [LibraryLoader.initialize()] → jedna rejestracja log callback → brak wyścigu.
///
/// ## Użycie
///
/// ```dart
/// final coordinator = await LlamaRagCoordinator.create(
///   embedModelPath: '/path/to/nomic-embed.gguf',
///   genModelPath: '/path/to/llama.gguf',
///   genConfig: LlmConfig(temp: 0.3, nCtx: 4096, nGpuLayers: 4),
/// );
///
/// final pipeline = RagPipeline(
///   embeddingProvider: coordinator.embeddingProvider,
///   vectorStore: InMemoryVectorStore(),
///   generationPlugin: coordinator.generationPlugin,
/// );
///
/// await coordinator.dispose();
/// ```
class LlamaRagCoordinator {
  Isolate? _isolate;
  SendPort? _workerPort;
  late final EmbeddingProvider _embeddingProvider;
  late final LlmPlugin _generationPlugin;

  LlamaRagCoordinator._();

  /// Tworzy koordynatora i ładuje oba modele w jednym worker isolate.
  ///
  /// [embedModelPath] — ścieżka do pliku .gguf modelu embeddingów
  /// [genModelPath]   — ścieżka do pliku .gguf modelu generowania
  /// [genConfig]      — konfiguracja modelu generowania
  /// [embedNCtx]      — rozmiar kontekstu embeddingów (domyślnie 512)
  ///
  /// Rzuca [FileSystemException] jeśli któryś z plików nie istnieje.
  /// Rzuca [Exception] jeśli ładowanie modelu nie powiodło się.
  static Future<LlamaRagCoordinator> create({
    required String embedModelPath,
    required String genModelPath,
    LlmConfig genConfig = const LlmConfig(),
    int embedNCtx = 512,
  }) async {
    if (!File(embedModelPath).existsSync()) {
      throw FileSystemException(
        'Model embeddingów nie istnieje',
        embedModelPath,
      );
    }
    if (!File(genModelPath).existsSync()) {
      throw FileSystemException(
        'Model generowania nie istnieje',
        genModelPath,
      );
    }

    final coordinator = LlamaRagCoordinator._();
    await coordinator._init(embedModelPath, genModelPath, genConfig, embedNCtx);
    return coordinator;
  }

  Future<void> _init(
    String embedModelPath,
    String genModelPath,
    LlmConfig genConfig,
    int embedNCtx,
  ) async {
    final initPort = ReceivePort();

    _isolate = await Isolate.spawn(
      _llamaRagWorkerMain,
      {
        'embedModelPath': embedModelPath,
        'genModelPath': genModelPath,
        'embedNCtx': embedNCtx,
        'genNCtx': genConfig.nCtxDefault,
        'nGpuLayers': genConfig.nGpuLayersDefault,
        'nBatch': genConfig.nBatchDefault,
        'nThreads': genConfig.nThreadsDefault,
        'temp': genConfig.tempDefault,
        'nPredict': genConfig.nPredictDefault,
        'topK': genConfig.topKDefault,
        'topP': genConfig.topPDefault,
        'penaltyRepeat': genConfig.penaltyRepeatDefault,
        'sendPort': initPort.sendPort,
      },
      debugName: 'llmcpp_RagWorker',
    );

    final initMsg = await initPort.first as Map<String, dynamic>;
    initPort.close();

    if (initMsg['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw Exception(
        'LlamaRagCoordinator init failed '
        '(${initMsg['phase']}): ${initMsg['message']}',
      );
    }

    _workerPort = initMsg['port'] as SendPort;

    // Sonduj wymiary embeddingów przez testowy embed
    final tempProvider = _CoordEmbeddingProvider(_workerPort!, dimensions: 0);
    final probeVec = await tempProvider.embed('dim_probe');

    _embeddingProvider = _CoordEmbeddingProvider(
      _workerPort!,
      dimensions: probeVec.length,
    );

    // Generation plugin — provider deleguje do worker isolate
    final genProvider = _CoordGenerationProvider(_workerPort!);
    _generationPlugin = LlmPlugin.custom(provider: genProvider, config: const {});
    await _generationPlugin.initialize(); // no-op w provider, ustawia isInitialized
  }

  /// Provider embeddingów — komunikuje się z worker isolate.
  EmbeddingProvider get embeddingProvider => _embeddingProvider;

  /// Plugin generowania — komunikuje się z worker isolate.
  LlmPlugin get generationPlugin => _generationPlugin;

  /// Czy koordynator jest gotowy do użycia.
  bool get isReady => _workerPort != null;

  /// Zwalnia oba modele i kończy worker isolate.
  Future<void> dispose() async {
    _workerPort?.send({'type': 'dispose'});
    await Future.delayed(const Duration(milliseconds: 200));
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _workerPort = null;
  }
}
