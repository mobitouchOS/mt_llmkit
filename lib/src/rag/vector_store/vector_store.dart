// lib/src/rag/vector_store/vector_store.dart

import 'dart:math' as math;

import '../document/document_chunk.dart';

/// Vector search result — a chunk with a similarity score.
class VectorSearchResult {
  /// Found document fragment
  final DocumentChunk chunk;

  /// Cosine similarity to the query (0.0–1.0, higher = better match)
  final double similarity;

  /// Position in the ranking (1-based)
  final int rank;

  const VectorSearchResult({
    required this.chunk,
    required this.similarity,
    required this.rank,
  });

  @override
  String toString() =>
      'VectorSearchResult(rank: $rank, similarity: ${similarity.toStringAsFixed(3)}, '
      'chunk: ${chunk.id})';
}

/// Abstract interface for a vector store.
///
/// Stores [DocumentChunk] instances with embeddings and enables
/// semantic search via cosine similarity.
///
/// ## Implementations
///
/// - [InMemoryVectorStore] — in-memory with automatic JSON persistence
abstract interface class VectorStore {
  /// Adds multiple chunks to the store.
  ///
  /// Chunks without [DocumentChunk.embedding] are ignored during search
  /// (but stored — they can be embedded later).
  Future<void> addChunks(List<DocumentChunk> chunks);

  /// Searches for chunks most similar to [queryEmbedding].
  ///
  /// [topK] — number of results to return (default 5)
  /// [minSimilarity] — minimum similarity (0.0–1.0), filters weak results
  Future<List<VectorSearchResult>> search(
    List<double> queryEmbedding, {
    int topK = 5,
    double minSimilarity = 0.0,
  });

  /// Removes all chunks belonging to the document with the given [documentId].
  Future<void> removeDocument(String documentId);

  /// Removes all chunks and clears persistence.
  Future<void> clear();

  /// Loads the vector store from a JSON file.
  Future<void> load(String path);

  /// Total number of chunks in the store (including those without embeddings)
  int get size;

  /// Number of chunks with a generated embedding (available for search)
  int get indexedSize;

  /// Unique document identifiers in the store
  List<String> get documentIds;
}

/// Utility: vector similarity computations.
///
/// Uses the proper cosine formula (does not assume normalised vectors),
/// which works correctly even if embeddings are not L2-normalised.
abstract final class VectorSimilarity {
  /// Cosine similarity between vectors [a] and [b].
  ///
  /// Returns a value in the range [-1.0, 1.0]:
  /// - 1.0 = identical directions (perfect match)
  /// - 0.0 = orthogonal (no relation)
  /// - -1.0 = opposite directions
  ///
  /// Returns 0.0 if either vector is zero or the dimensions do not match.
  static double cosine(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    final denom = math.sqrt(normA) * math.sqrt(normB);
    return denom == 0.0 ? 0.0 : dot / denom;
  }

  /// Euclidean distance between vectors [a] and [b].
  ///
  /// Smaller value = more similar. Alternative to cosine similarity.
  static double euclidean(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return double.infinity;
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return math.sqrt(sum);
  }
}
