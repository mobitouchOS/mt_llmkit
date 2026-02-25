// lib/src/rag/chunking/text_chunker.dart

import 'dart:math' as math;

import '../document/document.dart';
import '../document/document_chunk.dart';

/// Splits document text into overlapping fragments (chunks).
///
/// ## Sliding-window algorithm with intelligent splitting
///
/// 1. The window advances by `chunkSize - chunkOverlap` characters per iteration
/// 2. At each chunk boundary, a natural split point is sought
///    (sentence end: `.`, `!`, `?`, `\n`) in the last 1/3 of the window
/// 3. If none found — a word boundary (space) is used
/// 4. Chunks shorter than `minChunkSize` are skipped
///
/// ## Parameters
///
/// - `chunkSize` (default 500 chars ≈ 100–150 tokens) — target size
/// - `chunkOverlap` (default 100 chars) — overlap for context continuity
/// - `minChunkSize` (default 50 chars) — minimum chunk size, smaller ones are skipped
///
/// ## Example
///
/// ```dart
/// final chunker = TextChunker(chunkSize: 500, chunkOverlap: 100);
/// final chunks = chunker.chunk(document);
/// print('Split into ${chunks.length} chunks');
/// ```
class TextChunker {
  /// Target number of characters in a single chunk
  final int chunkSize;

  /// Number of overlapping characters between adjacent chunks
  final int chunkOverlap;

  /// Minimum number of characters for a chunk (smaller ones are skipped)
  final int minChunkSize;

  const TextChunker({
    this.chunkSize = 500,
    this.chunkOverlap = 100,
    this.minChunkSize = 50,
  }) : assert(
          chunkOverlap < chunkSize,
          'chunkOverlap must be less than chunkSize',
        );

  /// Splits [document] into a list of [DocumentChunk].
  ///
  /// Returns an empty list if the document is empty.
  List<DocumentChunk> chunk(Document document) {
    final text = document.content.trim();
    if (text.isEmpty) return [];

    final chunks = <DocumentChunk>[];
    int start = 0;
    int chunkIndex = 0;

    while (start < text.length) {
      final rawEnd = math.min(start + chunkSize, text.length);

      // Find a natural split point (do not do this at the end of the text)
      final end = rawEnd < text.length
          ? _findSplitPoint(text, start, rawEnd)
          : rawEnd;

      final chunkText = text.substring(start, end).trim();

      if (chunkText.length >= minChunkSize) {
        chunks.add(DocumentChunk(
          id: '${document.id}_chunk_$chunkIndex',
          documentId: document.id,
          text: chunkText,
          chunkIndex: chunkIndex,
          startChar: start,
          endChar: end,
          metadata: {
            'documentTitle': document.title,
            'documentSource': document.source,
          },
        ));
        chunkIndex++;
      }

      // Advance the window taking overlap into account
      final step = end - start - chunkOverlap;
      if (step <= 0) {
        // Guard against infinite loop for very small texts
        break;
      }
      start += step;
    }

    return chunks;
  }

  /// Finds the best split point in the range `[rawEnd * 2/3, rawEnd]`.
  ///
  /// Priority:
  /// 1. Sentence end (`.`, `!`, `?`) + space or newline
  /// 2. Newline (`\n`)
  /// 3. Word boundary (last space in the window)
  /// 4. Hard boundary `rawEnd` (fallback)
  int _findSplitPoint(String text, int start, int rawEnd) {
    // Search only in the last 1/3 of the window — preserve most of chunkSize
    final searchStart = start + (rawEnd - start) * 2 ~/ 3;
    final candidate = text.substring(searchStart, rawEnd);

    // 1. Sentence end followed by a space or newline
    final sentenceEnd = RegExp(r'[.!?][\s\n]');
    final sentenceMatch = sentenceEnd.allMatches(candidate).lastOrNull;
    if (sentenceMatch != null) {
      return searchStart + sentenceMatch.end;
    }

    // 2. Newline
    final newlineIdx = candidate.lastIndexOf('\n');
    if (newlineIdx >= 0) {
      return searchStart + newlineIdx + 1;
    }

    // 3. Last space (word boundary)
    final spaceIdx = candidate.lastIndexOf(' ');
    if (spaceIdx >= 0) {
      return searchStart + spaceIdx + 1;
    }

    // 4. Hard boundary
    return rawEnd;
  }
}
