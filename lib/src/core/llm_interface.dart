import 'streaming_result.dart';

abstract class LlmInterface {
  Future<void> loadModel(String localPath);
  Stream<String>? sendPrompt(String prompt);

  /// Sends a prompt and returns a stream of chunks with live performance metrics
  Stream<StreamingChunk> sendPromptStream(String prompt);

  void dispose();
  void clean();
}
