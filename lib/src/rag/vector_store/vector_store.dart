// lib/src/rag/vector_store/vector_store.dart

import 'dart:math' as math;

import '../document/document_chunk.dart';

/// Wynik wyszukiwania wektorowego — chunk z oceną podobieństwa.
class VectorSearchResult {
  /// Znaleziony fragment dokumentu
  final DocumentChunk chunk;

  /// Podobieństwo kosinusowe do zapytania (0.0–1.0, wyższe = bardziej pasuje)
  final double similarity;

  /// Pozycja w rankingu (1-based)
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

/// Abstrakcyjny interfejs bazy wektorów.
///
/// Przechowuje [DocumentChunk] z embeddingami i umożliwia wyszukiwanie
/// semantyczne przez podobieństwo kosinusowe.
///
/// ## Implementacje
///
/// - [InMemoryVectorStore] — in-memory z automatycznym zapisem JSON
abstract interface class VectorStore {
  /// Dodaje wiele chunków do bazy.
  ///
  /// Chunki bez [DocumentChunk.embedding] są ignorowane podczas wyszukiwania
  /// (ale przechowywane — mogą być zaembeddowane później).
  Future<void> addChunks(List<DocumentChunk> chunks);

  /// Wyszukuje najbardziej pasujące chunki do [queryEmbedding].
  ///
  /// [topK] — liczba wyników do zwrócenia (domyślnie 5)
  /// [minSimilarity] — minimalne podobieństwo (0.0–1.0), filtruje słabe wyniki
  Future<List<VectorSearchResult>> search(
    List<double> queryEmbedding, {
    int topK = 5,
    double minSimilarity = 0.0,
  });

  /// Usuwa wszystkie chunki należące do dokumentu o danym [documentId].
  Future<void> removeDocument(String documentId);

  /// Usuwa wszystkie chunki i czyści persistence.
  Future<void> clear();

  /// Ładuje bazę wektorów z pliku JSON.
  Future<void> load(String path);

  /// Czytelna liczba chunków w bazie (w tym bez embeddingów)
  int get size;

  /// Liczba chunków z wygenerowanym embeddingiem (dostępnych do wyszukiwania)
  int get indexedSize;

  /// Unikalne identyfikatory dokumentów w bazie
  List<String> get documentIds;
}

/// Utility: obliczenia podobieństwa wektorowego.
///
/// Używa właściwego wzoru kosinusowego (nie zakłada normalizacji wektorów),
/// co działa poprawnie nawet jeśli embeddingi nie są znormalizowane L2.
abstract final class VectorSimilarity {
  /// Podobieństwo kosinusowe między wektorami [a] i [b].
  ///
  /// Zwraca wartość w przedziale [-1.0, 1.0]:
  /// - 1.0 = identyczne kierunki (idealny match)
  /// - 0.0 = prostopadłe (brak związku)
  /// - -1.0 = przeciwne kierunki
  ///
  /// Zwraca 0.0 jeśli jeden z wektorów jest zerowy lub wymiarowości nie pasują.
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

  /// Odległość euklidesowa między wektorami [a] i [b].
  ///
  /// Mniejsza wartość = bardziej podobne. Alternatywa dla cosine similarity.
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
