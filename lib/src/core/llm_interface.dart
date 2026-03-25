import 'package:llamadart/llamadart.dart' show LlamaImageContent;

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

  /// Sends a prompt with image attachments and returns a stream of tokens.
  ///
  /// Requires the model to have been loaded with a multimodal projector
  /// (`mmprojPath` in [LlmConfig]). The prompt must contain one `<image>`
  /// placeholder for each image in [images].
  ///
  /// Throws [UnsupportedError] if vision is not configured.
  Stream<String> sendPromptWithImages(String prompt, List<LlamaImageContent> images);

  /// Sends a prompt with image attachments and waits for the complete response.
  ///
  /// Requires the model to have been loaded with a multimodal projector.
  /// Throws [UnsupportedError] if vision is not configured.
  Future<String> sendPromptCompleteWithImages(
    String prompt,
    List<LlamaImageContent> images,
  );

  /// Sends a prompt with image attachments and returns a stream of
  /// [StreamingChunk] with live performance metrics.
  ///
  /// Requires the model to have been loaded with a multimodal projector.
  /// Throws [UnsupportedError] if vision is not configured.
  Stream<StreamingChunk> sendPromptStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  );

  /// Whether generation is currently in progress.
  bool get isGenerating;

  /// Whether the model is loaded and ready for generation.
  bool get isInitialized;

  void dispose();
  void clean();
}
