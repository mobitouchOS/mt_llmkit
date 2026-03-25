import 'package:llamadart/llamadart.dart' show GpuBackend;

class LlmConfig {
  final int? nGpuLayers;
  final int? nCtx;
  final int? nBatch;
  final int? nPredict;
  final int? nThreads;
  final double? temp;
  final int? topK;
  final double? topP;
  final double? penaltyRepeat;

  /// GPU backend to use for inference. Defaults to [GpuBackend.auto] which
  /// tries Vulkan → Metal → CUDA → CPU in order. Use [GpuBackend.cpu] to
  /// disable GPU acceleration entirely (useful when Vulkan causes crashes).
  final GpuBackend? gpuBackend;

  /// Path to the multimodal projector GGUF file (e.g. `mmproj-model-f16.gguf`).
  ///
  /// Required when using vision models (LLaVA, Gemma 3, Qwen VL, etc.).
  /// When set, [GgufPlugin.sendPromptWithImages],
  /// [GgufPlugin.sendPromptCompleteWithImages], and
  /// [GgufPlugin.sendPromptStreamWithImages] become available.
  final String? mmprojPath;

  const LlmConfig({
    this.nGpuLayers,
    this.nCtx,
    this.nBatch,
    this.nPredict,
    this.nThreads,
    this.temp,
    this.topK,
    this.topP,
    this.penaltyRepeat,
    this.gpuBackend,
    this.mmprojPath,
  });

  int get nGpuLayersDefault => nGpuLayers ?? 64;
  int get nCtxDefault => nCtx ?? 8192;
  int get nBatchDefault => nBatch ?? 4096;
  int get nPredictDefault => nPredict ?? 8192;
  int get nThreadsDefault => nThreads ?? 6;
  double get tempDefault => temp ?? 0.72;
  int get topKDefault => topK ?? 64;
  double get topPDefault => topP ?? 0.95;
  double get penaltyRepeatDefault => penaltyRepeat ?? 1.1;
  // For android set cpu backend by default to avoid Vulkan crashes, but allow GPU acceleration on other platforms by default.
  GpuBackend get gpuBackendDefault => gpuBackend ?? GpuBackend.auto;
}
