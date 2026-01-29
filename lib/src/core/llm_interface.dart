abstract class LlmInterface {
  Future<void> loadModel(String localPath);
  Stream<String>? sendPrompt(String prompt);
  void dispose();
  void clean();
}
