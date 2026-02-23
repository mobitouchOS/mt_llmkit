// lib/src/rag/vector_store/in_memory_vector_store.dart

import 'dart:convert';
import 'dart:io';

import '../document/document_chunk.dart';
import 'vector_store.dart';

/// Implementacja [VectorStore] przechowująca chunki w pamięci RAM
/// z **automatycznym zapisem** do pliku JSON.
///
/// ## Persistence
///
/// - Przy starcie: wywołaj [load(path)] aby odtworzyć indeks z poprzedniej sesji
/// - Po każdym [addChunks] / [removeDocument] / [clear]: automatyczny zapis do [autoSavePath]
/// - Format: JSON z listą chunków (text + embedding + metadata)
///
/// ## Wyszukiwanie
///
/// Liniowe przeszukiwanie przez cosine similarity — odpowiednie dla
/// tysięcy chunków. Dla > 100k chunków rozważ implementację z HNSW/Annoy.
///
/// ## Przykład
///
/// ```dart
/// final store = InMemoryVectorStore();
///
/// // Wczytaj zapisany indeks (jeśli istnieje)
/// await store.load('/path/to/index.json');
///
/// // Dodaj chunki z embeddingami
/// await store.addChunks(chunks);  // automatyczny zapis po dodaniu
///
/// // Wyszukaj 5 najlepszych wyników
/// final results = await store.search(queryEmbedding, topK: 5, minSimilarity: 0.3);
/// ```
class InMemoryVectorStore implements VectorStore {
  final List<DocumentChunk> _chunks = [];

  /// Ścieżka do pliku JSON dla automatycznego zapisu.
  /// Ustawiana przez [load()] lub manualnie przed [addChunks()].
  String? autoSavePath;

  InMemoryVectorStore({this.autoSavePath});

  // ── VectorStore interface ──────────────────────────────────────────────

  @override
  Future<void> addChunks(List<DocumentChunk> chunks) async {
    _chunks.addAll(chunks);
    await _autoSave();
  }

  /// Wyszukuje [topK] chunków najbardziej podobnych do [queryEmbedding].
  ///
  /// Algorytm: cosine similarity przez [VectorSimilarity.cosine].
  /// Złożoność: O(n * d), gdzie n = liczba chunków, d = wymiarowość.
  @override
  Future<List<VectorSearchResult>> search(
    List<double> queryEmbedding, {
    int topK = 5,
    double minSimilarity = 0.0,
  }) async {
    final results = <VectorSearchResult>[];

    for (final chunk in _chunks) {
      if (chunk.embedding == null) continue;

      final similarity = VectorSimilarity.cosine(queryEmbedding, chunk.embedding!);
      if (similarity >= minSimilarity) {
        results.add(VectorSearchResult(
          chunk: chunk,
          similarity: similarity,
          rank: 0, // wypełniany poniżej
        ));
      }
    }

    // Sortuj malejąco po similarity
    results.sort((a, b) => b.similarity.compareTo(a.similarity));

    // Przypisz rankingi i ogranicz do topK
    final topResults = results.take(topK).toList();
    return List.generate(
      topResults.length,
      (i) => VectorSearchResult(
        chunk: topResults[i].chunk,
        similarity: topResults[i].similarity,
        rank: i + 1,
      ),
    );
  }

  @override
  Future<void> removeDocument(String documentId) async {
    _chunks.removeWhere((c) => c.documentId == documentId);
    await _autoSave();
  }

  @override
  Future<void> clear() async {
    _chunks.clear();
    // Usuń plik JSON jeśli istnieje
    if (autoSavePath != null) {
      final file = File(autoSavePath!);
      if (file.existsSync()) {
        await file.delete();
      }
    }
  }

  // ── Persistence ────────────────────────────────────────────────────────

  /// Wczytuje indeks z pliku JSON i ustawia [autoSavePath].
  ///
  /// Jeśli plik nie istnieje — baza pozostaje pusta (nie rzuca wyjątku).
  @override
  Future<void> load(String path) async {
    autoSavePath = path;
    final file = File(path);
    if (!file.existsSync()) return;

    try {
      final jsonString = await file.readAsString();
      final data = jsonDecode(jsonString) as Map<String, dynamic>;
      final rawChunks = data['chunks'] as List<dynamic>?;
      if (rawChunks == null) return;

      final loaded = rawChunks
          .map((e) => DocumentChunk.fromJson(e as Map<String, dynamic>))
          .toList();

      _chunks.addAll(loaded);
    } catch (e) {
      // Uszkodzony plik — zacznij z pustą bazą, nie przerywaj działania
      _chunks.clear();
    }
  }

  // ── Gettery ────────────────────────────────────────────────────────────

  @override
  int get size => _chunks.length;

  @override
  int get indexedSize => _chunks.where((c) => c.hasEmbedding).length;

  @override
  List<String> get documentIds =>
      _chunks.map((c) => c.documentId).toSet().toList();

  /// Wszystkie chunki (w tym bez embeddingów)
  List<DocumentChunk> get allChunks => List.unmodifiable(_chunks);

  // ── Prywatne helpers ───────────────────────────────────────────────────

  /// Zapisuje aktualny stan do [autoSavePath] jeśli jest ustawiony.
  Future<void> _autoSave() async {
    if (autoSavePath == null) return;
    await _persistTo(autoSavePath!);
  }

  Future<void> _persistTo(String path) async {
    final data = {
      'version': 1,
      'savedAt': DateTime.now().toIso8601String(),
      'chunkCount': _chunks.length,
      'indexedCount': indexedSize,
      'chunks': _chunks.map((c) => c.toJson()).toList(),
    };
    final file = File(path);
    await file.parent.create(recursive: true);
    await file.writeAsString(
      const JsonEncoder.withIndent(null).convert(data),
      flush: true,
    );
  }
}
