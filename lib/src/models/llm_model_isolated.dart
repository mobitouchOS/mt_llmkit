// lib/src/models/llm_model_isolated.dart
import 'dart:async';
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../llmcpp.dart';
import '../native/library_loader.dart';

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
  Stream<String> sendPrompt(String prompt) {
    checkInitialized();
    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    return _isolatedStream(formattedPrompt);
  }

  @override
  Future<String> sendPromptComplete(String prompt) async {
    checkInitialized();
    markGenerationStart();
    try {
      final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
      final buffer = StringBuffer();
      await for (final token in _promptTokenStream(formattedPrompt)) {
        buffer.write(token);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  // Wraps _promptTokenStream with isGenerating tracking for sendPrompt.
  Stream<String> _isolatedStream(String formattedPrompt) async* {
    markGenerationStart();
    try {
      yield* _promptTokenStream(formattedPrompt);
    } finally {
      markGenerationEnd();
    }
  }

  // Bridges the persistent broadcast stream to a per-prompt stream that closes
  // when waitForCompletion fires for this specific promptId.
  Stream<String> _promptTokenStream(String formattedPrompt) async* {
    final controller = StreamController<String>();
    final tokenSub = _llamaParent!.stream.listen(
      (token) { if (!controller.isClosed) controller.add(token); },
      onError: (Object e) { if (!controller.isClosed) controller.addError(e); },
    );

    final promptId = await _llamaParent!.sendPrompt(formattedPrompt);

    _llamaParent!.waitForCompletion(promptId).then((_) {
      tokenSub.cancel();
      if (!controller.isClosed) controller.close();
    }, onError: (_) {
      tokenSub.cancel();
      if (!controller.isClosed) controller.close();
    });

    try {
      await for (final token in controller.stream) {
        yield token;
      }
    } finally {
      await tokenSub.cancel();
      if (!controller.isClosed) await controller.close();
    }
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) async* {
    checkInitialized();

    final startTime = DateTime.now();
    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      await for (final chunk in _promptTokenStream(formattedPrompt)) {
        totalTokenCount += 1;

        yield StreamingChunk(
          text: chunk,
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
    } finally {
      markGenerationEnd();
    }
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
