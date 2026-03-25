import 'package:llamadart/llamadart.dart'
    show GpuBackend, LoraAdapterConfig, GenerationGrammarTrigger;

class LlmConfig {
  // ── ModelParams ────────────────────────────────────────────────────────────

  final int? nGpuLayers;
  final int? nCtx;
  final int? nBatch;
  final int? nThreads;

  /// Number of threads for batch processing (n_threads_batch). 0 = auto.
  final int? numberOfThreadsBatch;

  /// Physical micro-batch size (n_ubatch). 0 = defaults to batchSize.
  final int? microBatchSize;

  /// Maximum parallel sequence slots in context memory (n_seq_max).
  final int? maxParallelSequences;

  /// LoRA adapters to load with the model.
  final List<LoraAdapterConfig>? loras;

  /// Custom chat template to override the model's built-in template.
  final String? chatTemplate;

  /// GPU backend to use for inference. Defaults to [GpuBackend.auto] which
  /// tries Vulkan → Metal → CUDA → CPU in order. Use [GpuBackend.cpu] to
  /// disable GPU acceleration entirely (useful when Vulkan causes crashes).
  final GpuBackend? gpuBackend;

  /// Path to the multimodal projector GGUF file (e.g. `mmproj-model-f16.gguf`).
  ///
  /// Required when using vision models (LLaVA, Gemma 3, Qwen VL, etc.).
  /// When set, pass images via the `images` parameter of [LocalModel.sendPromptStream]
  /// and related methods to enable vision inference.
  final String? mmprojPath;

  // ── GenerationParams ───────────────────────────────────────────────────────

  final int? nPredict;
  final double? temp;
  final int? topK;
  final double? topP;

  /// Min-P sampling threshold. Set to 0.0 to disable.
  final double? minP;

  final double? penaltyRepeat;

  /// Random seed for the sampler. null = time-based seed.
  final int? seed;

  /// Strings that immediately stop generation when encountered.
  final List<String>? stopSequences;

  /// GBNF grammar string for structured output (e.g. `'root ::= "yes" | "no"'`).
  final String? grammar;

  /// Whether grammar should be lazily activated by [grammarTriggers].
  final bool? grammarLazy;

  /// Lazy grammar activation triggers. Used together with [grammarLazy].
  final List<GenerationGrammarTrigger>? grammarTriggers;

  /// Tokens to preserve during constrained decoding.
  final List<String>? preservedTokens;

  /// Grammar start symbol. Defaults to `'root'`.
  final String? grammarRoot;

  /// Reuse matching prompt prefixes from previous requests to reduce latency.
  final bool? reusePromptPrefix;

  /// Chunk flush threshold by token pieces (lower = finer stream granularity).
  final int? streamBatchTokenThreshold;

  /// Chunk flush threshold by byte size (lower = finer stream granularity).
  final int? streamBatchByteThreshold;

  const LlmConfig({
    this.nGpuLayers,
    this.nCtx,
    this.nBatch,
    this.nThreads,
    this.numberOfThreadsBatch,
    this.microBatchSize,
    this.maxParallelSequences,
    this.loras,
    this.chatTemplate,
    this.gpuBackend,
    this.mmprojPath,
    this.nPredict,
    this.temp,
    this.topK,
    this.topP,
    this.minP,
    this.penaltyRepeat,
    this.seed,
    this.stopSequences,
    this.grammar,
    this.grammarLazy,
    this.grammarTriggers,
    this.preservedTokens,
    this.grammarRoot,
    this.reusePromptPrefix,
    this.streamBatchTokenThreshold,
    this.streamBatchByteThreshold,
  });

  // ModelParams defaults
  int get nGpuLayersDefault => nGpuLayers ?? 64;
  int get nCtxDefault => nCtx ?? 8192;
  int get nBatchDefault => nBatch ?? 4096;
  int get nThreadsDefault => nThreads ?? 6;
  int get numberOfThreadsBatchDefault => numberOfThreadsBatch ?? 0;
  int get microBatchSizeDefault => microBatchSize ?? 0;
  int get maxParallelSequencesDefault => maxParallelSequences ?? 1;
  List<LoraAdapterConfig> get lorasDefault => loras ?? const [];
  GpuBackend get gpuBackendDefault => gpuBackend ?? GpuBackend.auto;

  // GenerationParams defaults
  int get nPredictDefault => nPredict ?? 8192;
  double get tempDefault => temp ?? 0.72;
  int get topKDefault => topK ?? 64;
  double get topPDefault => topP ?? 0.95;
  double get minPDefault => minP ?? 0.0;
  double get penaltyRepeatDefault => penaltyRepeat ?? 1.1;
  List<String> get stopSequencesDefault => stopSequences ?? const [];
  bool get grammarLazyDefault => grammarLazy ?? false;
  List<GenerationGrammarTrigger> get grammarTriggersDefault =>
      grammarTriggers ?? const [];
  List<String> get preservedTokensDefault => preservedTokens ?? const [];
  String get grammarRootDefault => grammarRoot ?? 'root';
  bool get reusePromptPrefixDefault => reusePromptPrefix ?? true;
  int get streamBatchTokenThresholdDefault => streamBatchTokenThreshold ?? 8;
  int get streamBatchByteThresholdDefault => streamBatchByteThreshold ?? 512;
}
