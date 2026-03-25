// lib/src/models/llm_model_standard.dart
import 'dart:async';
import 'dart:io';

import 'package:llamadart/llamadart.dart';

import '../../llmcpp.dart';

class LlmModelStandard extends LlmModelBase {
  final LlmConfig config;
  LlamaEngine? _engine;

  LlmModelStandard(this.config);

  ModelParams get _modelParams => ModelParams(
    contextSize: config.nCtxDefault,
    gpuLayers: config.nGpuLayersDefault,
    batchSize: config.nBatchDefault,
    numberOfThreads: config.nThreadsDefault,
    preferredBackend: config.gpuBackendDefault,
  );

  GenerationParams get _genParams => GenerationParams(
    maxTokens: config.nPredictDefault,
    temp: config.tempDefault,
    topK: config.topKDefault,
    topP: config.topPDefault,
    penalty: config.penaltyRepeatDefault,
  );

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    _engine = LlamaEngine(LlamaBackend());
    await _engine!.loadModel(localPath, modelParams: _modelParams);

    if (config.mmprojPath != null) {
      await _engine!.loadMultimodalProjector(config.mmprojPath!);
    }

    markAsInitialized();
  }

  @override
  Stream<String> sendPrompt(String prompt) {
    checkInitialized();
    return _bufferedStream(prompt);
  }

  @override
  Future<String> sendPromptComplete(String prompt) async {
    checkInitialized();
    markGenerationStart();
    try {
      final buffer = StringBuffer();
      await for (final chunk in _engine!.create(
        [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text != null) buffer.write(text);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  Stream<String> _bufferedStream(String prompt) async* {
    if (_engine == null) return;

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    markGenerationStart();
    try {
      await for (final chunk in _engine!.create(
        [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text == null) continue;

        buffer.write(text);

        if (stopwatch.elapsed >= yieldInterval) {
          if (buffer.isNotEmpty) {
            yield buffer.toString();
            buffer.clear();
          }
          stopwatch.reset();
          await Future.delayed(Duration.zero);
        }
      }

      if (buffer.isNotEmpty) {
        yield buffer.toString();
      }
    } finally {
      markGenerationEnd();
    }
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) async* {
    checkInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      await for (final chunk in _engine!.create(
        [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text == null) continue;

        totalTokenCount += 1;

        yield StreamingChunk(
          text: text,
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
  Stream<String> sendPromptWithImages(String prompt, List<LlamaImageContent> images) {
    checkInitialized();
    return _bufferedStreamWithImages(prompt, images);
  }

  @override
  Future<String> sendPromptCompleteWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) async {
    checkInitialized();
    markGenerationStart();
    try {
      final buffer = StringBuffer();
      await for (final chunk in _engine!.create(
        [
          LlamaChatMessage.withContent(
            role: LlamaChatRole.user,
            content: [LlamaTextContent(prompt), ...images],
          ),
        ],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text != null) buffer.write(text);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  @override
  Stream<StreamingChunk> sendPromptStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) async* {
    checkInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      await for (final chunk in _engine!.create(
        [
          LlamaChatMessage.withContent(
            role: LlamaChatRole.user,
            content: [LlamaTextContent(prompt), ...images],
          ),
        ],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text == null) continue;

        totalTokenCount += 1;

        yield StreamingChunk(
          text: text,
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

  Stream<String> _bufferedStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) async* {
    if (_engine == null) return;

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    markGenerationStart();
    try {
      await for (final chunk in _engine!.create(
        [
          LlamaChatMessage.withContent(
            role: LlamaChatRole.user,
            content: [LlamaTextContent(prompt), ...images],
          ),
        ],
        params: _genParams,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text == null) continue;

        buffer.write(text);

        if (stopwatch.elapsed >= yieldInterval) {
          if (buffer.isNotEmpty) {
            yield buffer.toString();
            buffer.clear();
          }
          stopwatch.reset();
          await Future.delayed(Duration.zero);
        }
      }

      if (buffer.isNotEmpty) {
        yield buffer.toString();
      }
    } finally {
      markGenerationEnd();
    }
  }

  @override
  void dispose() {
    _engine?.dispose(); // Future<void> — fire-and-forget
    _engine = null;
    markAsDisposed();
  }

  @override
  void clean() {
    checkInitialized();
    // create() is stateless in llamadart — no context to reset
  }
}
