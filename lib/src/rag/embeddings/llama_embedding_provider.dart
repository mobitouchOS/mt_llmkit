// lib/src/rag/embeddings/llama_embedding_provider.dart

import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../../core/llm_config.dart';
import 'embedding_provider.dart';

// ── Worker Isolate entry point ─────────────────────────────────────────────

Future<void> _embeddingWorkerMain(Map<String, dynamic> args) async {
  final String modelPath = args['modelPath'] as String;
  final int contextSize = args['contextSize'] as int;
  final int batchSize = args['batchSize'] as int;
  final int numberOfThreads = args['numberOfThreads'] as int;
  final SendPort mainPort = args['sendPort'] as SendPort;

  final engine = LlamaEngine(LlamaBackend());
  try {
    await engine.loadModel(
      modelPath,
      modelParams: ModelParams(
        contextSize: contextSize,
        gpuLayers: 0,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        preferredBackend: GpuBackend.cpu,
      ),
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'message': '$e'});
    return;
  }

  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          final embedding = await engine.embed(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', 'message': '$e'});
        }

      case 'dispose':
        await engine.dispose();
        receivePort.close();
        return;
    }
  }
}

// ── LlamaEmbeddingProvider ─────────────────────────────────────────────────

class LlamaEmbeddingProvider implements EmbeddingProvider {
  Isolate? _isolate;
  SendPort? _workerPort;
  int _dimensions = 0;
  bool _isInitialized = false;

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

    final initReceivePort = ReceivePort();
    _isolate = await Isolate.spawn(
      _embeddingWorkerMain,
      {
        'modelPath': modelPath,
        'contextSize': (config['nCtx'] as int?) ?? 512,
        'batchSize': llmConfig.nBatchDefault,
        'numberOfThreads': llmConfig.nThreadsDefault,
        'sendPort': initReceivePort.sendPort,
      },
      debugName: 'llmcpp_EmbeddingWorker',
    );

    final initResponse = await initReceivePort.first as Map<String, dynamic>;
    initReceivePort.close();

    if (initResponse['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw Exception('Failed to load embedding model: ${initResponse['message']}');
    }

    _workerPort = initResponse['port'] as SendPort;
    final testVec = await embed('dimension_probe');
    _dimensions = testVec.length;
    _isInitialized = true;
  }

  @override
  Future<List<double>> embed(String text) async {
    if (_workerPort == null) {
      throw StateError('EmbeddingProvider is not initialized. Call initialize() first.');
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
    _workerPort?.send({'type': 'dispose'});
    await Future.delayed(const Duration(milliseconds: 100));
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

  String _truncateText(String text, {int maxChars = 2000}) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(' ');
    return lastSpace > maxChars * 0.8 ? truncated.substring(0, lastSpace) : truncated;
  }
}
