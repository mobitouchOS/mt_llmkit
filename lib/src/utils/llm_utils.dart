import 'package:llama_cpp_dart/llama_cpp_dart.dart';

abstract class LlmUtils {
  /// Counts tokens using the model's real BPE tokenizer when a [Llama]
  /// instance is available, or falls back to a word + punctuation heuristic
  /// when [llama] is null (e.g. in the isolate backend or RAG coordinator
  /// where direct vocabulary access is not possible).
  ///
  /// Pass [addBos] = false (default) when counting individual streaming chunks
  /// so that the BOS token is not included in every chunk's count.
  static int tokenizerEstimateTokens(String text, Llama? llama,
      {bool addBos = false}) {
    if (text.isEmpty) return 0;

    if (llama != null) {
      return llama.tokenize(text, addBos).length;
    }

    // Fallback heuristic: word + punctuation counting.
    // Used when the tokenizer is unavailable (LlamaParent / LlmInterface).
    final wordCount = RegExp(r'\w+').allMatches(text).length;
    var punctuationCount = 0;
    for (final mark in const [
      ',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}',
    ]) {
      punctuationCount += mark.allMatches(text).length;
    }
    return wordCount + punctuationCount;
  }
}
