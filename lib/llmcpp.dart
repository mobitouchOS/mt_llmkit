import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

export 'package:llama_cpp_dart/llama_cpp_dart.dart'
    show PromptFormat, ChatMLFormat, AlpacaFormat, GemmaFormat;
export 'package:llama_cpp_dart/src/harmony_format.dart';
export 'package:llama_cpp_dart/src/prompt_format.dart';

class Llmcpp {
  final PromptFormat? promptFormat;
  // export promptFormat classes

  /// Set 0 for simulators
  final int? nGpuLayers;
  final int? nCtx;
  final int? nBatch;
  final int? nPredict;
  final int? nThreads;
  final double? temp;
  final int? topK;
  final double? topP;
  final double? penaltyRepeat;

  Llmcpp({
    this.promptFormat,
    this.nGpuLayers,
    this.nCtx,
    this.nBatch,
    this.nPredict,
    this.nThreads,
    this.temp,
    this.topK,
    this.topP,
    this.penaltyRepeat,
  }) {
    _initLibs();
  }

  // Values with defaults
  int get _nGpuLayersDefault => nGpuLayers ?? 64;
  int get _nCtxDefault => nCtx ?? 8192;
  int get _nBatchDefault => nBatch ?? 4096;
  int get _nPredictDefault => nPredict ?? 8192;
  int get _nThreadsDefault => nThreads ?? 6;
  double get _tempDefault => temp ?? 0.72;
  int get _topKDefault => topK ?? 64;
  double get _topPDefault => topP ?? 0.95;
  double get _penaltyRepeatDefault => penaltyRepeat ?? 1.1;
  PromptFormat get _promptFormatDefault => promptFormat ?? ChatMLFormat();

  Llama? llama;
  final _ctx = ContextParams();
  int get _nCtx => _ctx.nCtx;
  late final model = llama;
  LlamaParent? llamaParent;

  void dispose() {
    llama?.dispose();
    llamaParent?.dispose();
  }

  void clean() {
    llama?.clear();
  }

  void _initLibs() {
    if (Platform.isAndroid) {
      // images?
      // Llama.libraryPath = 'libmtmd.so';
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isLinux) {
      try {
        final libs = [
          'libggml-base.so',
          'libggml-cpu.so',
          'libggml-opencl.so',
          'libggml-vulkan.so',
          'libggml.so',
          'libllama.so',
          'libmtmd.so',
        ];
        for (final lib in libs) {
          ffi.DynamicLibrary.open(lib);
        }
      } catch (e) {
        if (kDebugMode) print('ggml preload failed (continuing): $e');
      }
      Llama.libraryPath = 'libmtmd.so';
      //       Llama.libraryPath = 'libllama.so';
      //     } else if (Platform.isIOS || Platform.isMacOS) {
      //       // iOS/macOS uses xcframework, no need to set path
      //       if (kDebugMode) print('Using bundled llama.xcframework');
    }
  }

  Stream<String>? sendPromptIsolated(String prompt) {
    if (llamaParent == null) {
      return null;
    }

    final formattedPrompt = _promptFormatDefault.formatPrompt(prompt);

    llamaParent!.sendPrompt(formattedPrompt);

    return llamaParent!.stream;
  }

  Future<void> loadIsolated(String localPath) async {
    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    try {
      final loadCommand = LlamaLoad(
        path: localPath,
        modelParams: ModelParams()..nGpuLayers = _nGpuLayersDefault,
        contextParams: ContextParams()
          ..nCtx = _nCtxDefault
          ..nBatch = _nBatchDefault
          ..nPredict = _nPredictDefault
          ..nThreads = _nThreadsDefault,
        samplingParams: SamplerParams()
          ..temp = _tempDefault
          ..topK = _topKDefault
          ..topP = _topPDefault
          ..penaltyRepeat = _penaltyRepeatDefault,
      );

      llamaParent = LlamaParent(loadCommand);
      await llamaParent?.init();
    } catch (e, s) {
      print('Error in _loadIsolated: $e');
    }
  }

  Future<void> loadModel(String localPath) async {
    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    final modelParams = ModelParams()..nGpuLayers = _nGpuLayersDefault;

    final contextParams = ContextParams()
      ..nPredict = _nPredictDefault
      ..nCtx = _nCtxDefault
      ..nBatch = _nBatchDefault
      ..nThreads = _nThreadsDefault;
    _ctx.nCtx = contextParams.nCtx;
    final samplerParams = SamplerParams()
      ..temp = _tempDefault
      ..topK = _topKDefault
      ..topP = _topPDefault
      ..penaltyRepeat = _penaltyRepeatDefault;

    final localMMProjPath = null;
    try {
      llama = Llama(
        localPath,
        // modelParams,
        // contextParams,
        // samplerParams,
        modelParams: modelParams,
        contextParams: contextParams,
        samplerParams: samplerParams,
      );
    } catch (e, stack) {
      print(stack);
      print(e);
      llama = null;
      return;
    }

    try {
      final remaining = llama!.getRemainingContextSpace();
      final used = (_nCtx - remaining).clamp(0, _nCtx);
    } catch (_) {}
  }

  Stream<String>? sendPrompt(String prompt) async* {
    if (llama == null || _promptFormatDefault == null) {
      return;
    }

    final formattedPrompt = _promptFormatDefault!.formatPrompt(prompt);

    llama!.setPrompt(formattedPrompt);

    // return llama!.generateText();

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    await for (final token in llama!.generateText()) {
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
}
