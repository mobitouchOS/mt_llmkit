// lib/src/models/llm_model_isolated.dart
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
  Stream<String>? sendPrompt(String prompt) {
    checkInitialized();

    final formattedPrompt = config.promptFormatDefault.formatPrompt(prompt);
    _llamaParent!.sendPrompt(formattedPrompt);

    return _llamaParent!.stream;
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
