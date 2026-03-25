// lib/src/rag/llama_rag_coordinator.dart
//
// ARCHITECTURE — single isolate for both models
//
// llamadart resolves the NativeCallable isolate-crash issue internally.
// We still use one isolate for both models to:
// 1. Keep the UI thread free during inference
// 2. Avoid any residual resource contention between two LlamaEngine instances

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../core/llm_config.dart';
import '../core/llm_interface.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'embeddings/embedding_provider.dart';

// ── Worker isolate ────────────────────────────────────────────────────────────

Future<void> _llamaRagWorkerMain(Map<String, dynamic> args) async {
  final String embedModelPath = args['embedModelPath'] as String;
  final String genModelPath = args['genModelPath'] as String;
  final int embedContextSize = args['embedContextSize'] as int;
  final int genContextSize = args['genContextSize'] as int;
  final int batchSize = args['batchSize'] as int;
  final int numberOfThreads = args['numberOfThreads'] as int;
  final int maxTokens = args['maxTokens'] as int;
  final double temp = (args['temp'] as num).toDouble();
  final int topK = args['topK'] as int;
  final double topP = (args['topP'] as num).toDouble();
  final double penalty = (args['penalty'] as num).toDouble();
  final String gpuBackendName = args['gpuBackend'] as String;
  final GpuBackend genGpuBackend = GpuBackend.values.firstWhere(
    (e) => e.name == gpuBackendName,
    orElse: () => GpuBackend.auto,
  );
  final SendPort mainPort = args['sendPort'] as SendPort;

  final embedEngine = LlamaEngine(LlamaBackend());
  try {
    await embedEngine.loadModel(
      embedModelPath,
      modelParams: ModelParams(
        contextSize: embedContextSize,
        gpuLayers: 0,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        preferredBackend: GpuBackend.cpu,
      ),
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'phase': 'embed_init', 'message': '$e'});
    return;
  }

  final genEngine = LlamaEngine(LlamaBackend());
  try {
    await genEngine.loadModel(
      genModelPath,
      modelParams: ModelParams(
        contextSize: genContextSize,
        gpuLayers: 0,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        preferredBackend: genGpuBackend,
      ),
    );
  } catch (e) {
    await embedEngine.dispose();
    mainPort.send({'type': 'error', 'phase': 'gen_init', 'message': '$e'});
    return;
  }

  final genParams = GenerationParams(
    maxTokens: maxTokens,
    temp: temp,
    topK: topK,
    topP: topP,
    penalty: penalty,
  );

  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  StreamSubscription<LlamaCompletionChunk>? genSubscription;

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          final embedding = await embedEngine.embed(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', 'message': '$e'});
        }

      case 'generate':
        final prompt = message['prompt'] as String;
        final streamPort = message['streamPort'] as SendPort;
        genSubscription = genEngine
            .create(
              [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
              params: genParams,
            )
            .listen(
              (chunk) {
                final text = chunk.choices.firstOrNull?.delta.content;
                if (text != null) streamPort.send({'type': 'token', 'text': text});
              },
              onDone: () {
                streamPort.send({'type': 'done'});
                genSubscription = null;
              },
              onError: (Object e) {
                streamPort.send({'type': 'error', 'message': '$e'});
                genSubscription = null;
              },
            );

      case 'stop_generate':
        genSubscription?.cancel();
        genSubscription = null;
        genEngine.cancelGeneration();

      case 'dispose':
        genSubscription?.cancel();
        await embedEngine.dispose();
        await genEngine.dispose();
        receivePort.close();
        return;
    }
  }
}

// ── Private provider implementations ─────────────────────────────────────────

class _CoordEmbeddingProvider implements EmbeddingProvider {
  final SendPort _workerPort;
  final int _dimensions;
  bool _isInitialized = true;

  _CoordEmbeddingProvider(this._workerPort, {required int dimensions})
      : _dimensions = dimensions;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    _isInitialized = true;
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
      throw Exception('Embedding error: ${response['message']}');
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
    return lastSpace > maxChars * 0.8 ? truncated.substring(0, lastSpace) : truncated;
  }
}

class _CoordPlugin implements LlmInterface {
  final SendPort _workerPort;
  bool _isGenerating = false;

  _CoordPlugin(this._workerPort);

  @override
  bool get isInitialized => true;

  @override
  bool get isGenerating => _isGenerating;

  @override
  Future<void> loadModel(String localPath) async {}

  @override
  Stream<String> sendPrompt(String prompt) {
    final controller = StreamController<String>();
    final replyPort = ReceivePort();

    _workerPort.send({
      'type': 'generate',
      'prompt': prompt,
      'streamPort': replyPort.sendPort,
    });

    _isGenerating = true;
    final sub = replyPort.listen((dynamic message) {
      if (message is! Map<String, dynamic>) return;
      switch (message['type'] as String?) {
        case 'token':
          if (!controller.isClosed) controller.add(message['text'] as String);
        case 'done':
          replyPort.close();
          _isGenerating = false;
          if (!controller.isClosed) controller.close();
        case 'error':
          replyPort.close();
          _isGenerating = false;
          if (!controller.isClosed) {
            controller.addError(Exception('Generation error: ${message['message']}'));
          }
      }
    });

    controller.onCancel = () {
      sub.cancel();
      replyPort.close();
      _isGenerating = false;
      _workerPort.send({'type': 'stop_generate'});
    };

    return controller.stream;
  }

  @override
  Future<String> sendPromptComplete(String prompt) async {
    final buffer = StringBuffer();
    await for (final token in sendPrompt(prompt)) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) async* {
    final startTime = DateTime.now();
    int totalTokenCount = 0;

    await for (final token in sendPrompt(prompt)) {
      totalTokenCount += 1;
      yield StreamingChunk(
        text: token,
        metrics: PerformanceMetrics.fromGeneration(
          tokenCount: totalTokenCount,
          startTime: startTime,
          endTime: DateTime.now(),
        ),
        isFinal: false,
      );
    }

    yield StreamingChunk(
      text: '',
      metrics: PerformanceMetrics.fromGeneration(
        tokenCount: totalTokenCount,
        startTime: startTime,
        endTime: DateTime.now(),
      ),
      isFinal: true,
    );
  }

  @override
  void dispose() {}

  @override
  void clean() {}

  @override
  Stream<String> sendPromptWithImages(String prompt, List<LlamaImageContent> images) {
    throw UnsupportedError('Vision is not supported in the RAG pipeline.');
  }

  @override
  Future<String> sendPromptCompleteWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    throw UnsupportedError('Vision is not supported in the RAG pipeline.');
  }

  @override
  Stream<StreamingChunk> sendPromptStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    throw UnsupportedError('Vision is not supported in the RAG pipeline.');
  }
}

// ── LlamaRagCoordinator ───────────────────────────────────────────────────────

class LlamaRagCoordinator {
  Isolate? _isolate;
  SendPort? _workerPort;
  late final EmbeddingProvider _embeddingProvider;
  late final LlmInterface _generationPlugin;

  LlamaRagCoordinator._();

  static Future<LlamaRagCoordinator> create({
    required String embedModelPath,
    required String genModelPath,
    LlmConfig genConfig = const LlmConfig(),
    int embedNCtx = 512,
  }) async {
    if (!File(embedModelPath).existsSync()) {
      throw FileSystemException('Embedding model does not exist', embedModelPath);
    }
    if (!File(genModelPath).existsSync()) {
      throw FileSystemException('Generation model does not exist', genModelPath);
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
        'embedContextSize': embedNCtx,
        'genContextSize': genConfig.nCtxDefault,
        'batchSize': genConfig.nBatchDefault,
        'numberOfThreads': genConfig.nThreadsDefault,
        'maxTokens': genConfig.nPredictDefault,
        'temp': genConfig.tempDefault,
        'topK': genConfig.topKDefault,
        'topP': genConfig.topPDefault,
        'penalty': genConfig.penaltyRepeatDefault,
        'gpuBackend': genConfig.gpuBackendDefault.name,
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
        'LlamaRagCoordinator init failed (${initMsg['phase']}): ${initMsg['message']}',
      );
    }

    _workerPort = initMsg['port'] as SendPort;

    final tempProvider = _CoordEmbeddingProvider(_workerPort!, dimensions: 0);
    final probeVec = await tempProvider.embed('dim_probe');

    _embeddingProvider = _CoordEmbeddingProvider(_workerPort!, dimensions: probeVec.length);
    _generationPlugin = _CoordPlugin(_workerPort!);
  }

  EmbeddingProvider get embeddingProvider => _embeddingProvider;
  LlmInterface get generationPlugin => _generationPlugin;
  bool get isReady => _workerPort != null;

  Future<void> dispose() async {
    _workerPort?.send({'type': 'dispose'});
    await Future.delayed(const Duration(milliseconds: 200));
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _workerPort = null;
  }
}
