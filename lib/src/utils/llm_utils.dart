abstract class LlmUtils {
  /// Estimates token count in a text chunk
  /// This is an approximation - real tokenization depends on the model
  static int estimateTokenCount(String text) {
    if (text.isEmpty) return 0;

    // Simple heuristic: count words and punctuation
    // Most LLMs use approximately 1 token per word + separate tokens for punctuation

    // Count words (sequences of letters/numbers)
    final wordMatches = RegExp(r'\w+').allMatches(text);
    final wordCount = wordMatches.length;

    // Count significant punctuation marks (each is usually a separate token)
    final punctuationMarks = [
      ',',
      '.',
      '!',
      '?',
      ';',
      ':',
      '"',
      "'",
      '(',
      ')',
      '[',
      ']',
      '{',
      '}',
    ];
    var punctuationCount = 0;
    for (final mark in punctuationMarks) {
      punctuationCount += mark.allMatches(text).length;
    }

    // Return total estimated tokens
    return wordCount + punctuationCount;
  }
}
