import 'streaming_result.dart';

abstract interface class LlmInterface {
  Future<void> loadModel(String localPath);

  /// Sends a prompt and returns a stream of tokens.
  ///
  /// Use [isGenerating] to check if generation is still in progress.
  Stream<String> sendPrompt(String prompt);

  /// Sends a prompt and waits for the complete response.
  Future<String> sendPromptComplete(String prompt);

  /// Sends a prompt and returns a stream of [StreamingChunk] with live
  /// performance metrics.
  Stream<StreamingChunk> sendPromptStream(String prompt);

  /// Whether generation is currently in progress.
  bool get isGenerating;

  /// Whether the model is loaded and ready for generation.
  bool get isInitialized;

  void dispose();
  void clean();
}
