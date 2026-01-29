import '../../llmcpp.dart';

class LlmConfig {
  final PromptFormat? promptFormat;
  final int? nGpuLayers;
  final int? nCtx;
  final int? nBatch;
  final int? nPredict;
  final int? nThreads;
  final double? temp;
  final int? topK;
  final double? topP;
  final double? penaltyRepeat;

  const LlmConfig({
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
  PromptFormat get promptFormatDefault => promptFormat ?? ChatMLFormat();
}
