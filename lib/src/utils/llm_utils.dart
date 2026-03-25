abstract class LlmUtils {
  /// Estimates the token count for [text] using a word + punctuation heuristic.
  ///
  /// This is an approximation (~75–90% accuracy) used when the tokenizer is
  /// unavailable (e.g. in isolate backends or the RAG coordinator).
  static int tokenizerEstimateTokens(String text) {
    if (text.isEmpty) return 0;

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
