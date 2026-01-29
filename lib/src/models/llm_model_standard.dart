// lib/src/models/llm_model_standard.dart
import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

import '../../llmcpp.dart';
import '../native/library_loader.dart';

class LlmModelStandard extends LlmModelBase {
  final LlmConfig config;
  Llama? _llama;

  LlmModelStandard(this.config);

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    LibraryLoader.initialize();

    _llama = Llama(
      localPath,
      modelParams: ModelParams()..nGpuLayers = config.nGpuLayersDefault,
      contextParams: ContextParams()
        ..nPredict = config.nPredictDefault
        ..nCtx = config.nCtxDefault
        ..nBatch = config.nBatchDefault
        ..nThreads = config.nThreadsDefault,
      samplerParams: SamplerParams()
        ..temp = config.tempDefault
        ..topK = config.topKDefault
        ..topP = config.topPDefault
        ..penaltyRepeat = config.penaltyRepeatDefault,
    );

    markAsInitialized();
  }

  @override
  Stream<String>? sendPrompt(String prompt) {
    checkInitialized();

    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    _llama!.setPrompt(formattedPrompt);

    return _llamaBufferedStream();
  }

  Stream<String> _llamaBufferedStream() async* {
    if (_llama == null) return;

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    await for (final token in _llama!.generateText()) {
      buffer.write(token);

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
  }

  @override
  void dispose() {
    _llama?.dispose();
    _llama = null;
    markAsDisposed();
  }

  @override
  void clean() {
    checkInitialized();
    _llama?.clear();
  }
}
