// lib/src/models/llm_model_isolated.dart
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../llmcpp.dart';
import '../native/library_loader.dart';
import '../utils/llm_utils.dart';

class LlmModelIsolated extends LlmModelBase {
  final LlmConfig config;
  LlamaParent? _llamaParent;

  LlmModelIsolated(this.config) {
    LibraryLoader.initialize();
  }

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    final loadCommand = LlamaLoad(
      path: localPath,
      verbose: true,
      modelParams: ModelParams()
        ..nGpuLayers = config.nGpuLayersDefault
        ..mainGpu = -1,
      contextParams: ContextParams()
        ..nCtx = config.nCtxDefault
        ..nBatch = config.nBatchDefault
        ..nPredict = config.nPredictDefault
        ..nThreads = config.nThreadsDefault,
      samplingParams: SamplerParams()
        ..temp = config.tempDefault
        ..topK = config.topKDefault
        ..topP = config.topPDefault
        ..penaltyRepeat = config.penaltyRepeatDefault,
    );

    _llamaParent = LlamaParent(loadCommand);
    await _llamaParent?.init();

    markAsInitialized();
  }

  @override
  Stream<String>? sendPrompt(String prompt) {
    checkInitialized();

    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    _llamaParent!.sendPrompt(formattedPrompt);

    return _llamaParent!.stream;
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) async* {
    checkInitialized();

    final startTime = DateTime.now();
    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    int totalTokenCount = 0;

    _llamaParent!.sendPrompt(formattedPrompt);

    await for (final chunk in _llamaParent!.stream) {
      // Count actual tokens in the chunk (approximate: split by whitespace and punctuation)
      // Each chunk may contain multiple tokens
      final tokensInChunk = LlmUtils.estimateTokenCount(chunk);
      totalTokenCount += tokensInChunk;

      final currentTime = DateTime.now();

      // Calculate current metrics
      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: totalTokenCount,
        startTime: startTime,
        endTime: currentTime,
      );

      yield StreamingChunk(text: chunk, metrics: metrics, isFinal: false);
    }

    // Send final chunk with final metrics
    final endTime = DateTime.now();
    final finalMetrics = PerformanceMetrics.fromGeneration(
      tokenCount: totalTokenCount,
      startTime: startTime,
      endTime: endTime,
    );

    yield StreamingChunk(text: '', metrics: finalMetrics, isFinal: true);
  }

  @override
  void dispose() {
    _llamaParent?.dispose();
    _llamaParent = null;
    markAsDisposed();
  }

  @override
  void clean() {
    checkInitialized();
  }
}
