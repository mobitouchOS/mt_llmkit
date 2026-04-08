// lib/src/rag/vector_store/in_memory_vector_store.dart

import 'dart:convert';
import 'dart:io';

import '../document/document_chunk.dart';
import 'vector_store.dart';

/// Implementation of [VectorStore] that stores chunks in RAM
/// with **automatic persistence** to a JSON file.
///
/// ## Persistence
///
/// - At startup: call [load(path)] to restore the index from a previous session
/// - After each [addChunks] / [removeDocument] / [clear]: auto-save to [autoSavePath]
/// - Format: JSON with a list of chunks (text + embedding + metadata)
///
/// ## Search
///
/// Linear scan via cosine similarity — suitable for
/// thousands of chunks. For > 100k chunks consider an HNSW/Annoy implementation.
///
/// ## Example
///
/// ```dart
/// final store = InMemoryVectorStore();
///
/// // Load a saved index (if one exists)
/// await store.load('/path/to/index.json');
///
/// // Add chunks with embeddings
/// await store.addChunks(chunks);  // auto-saved after adding
///
/// // Search for the 5 best results
/// final results = await store.search(queryEmbedding, topK: 5, minSimilarity: 0.3);
/// ```
class InMemoryVectorStore implements VectorStore {
  final List<DocumentChunk> _chunks = [];

  /// Path to the JSON file for automatic saving.
  /// Set by [load()] or manually before [addChunks()].
  String? autoSavePath;

  InMemoryVectorStore({this.autoSavePath});

  // ── VectorStore interface ──────────────────────────────────────────────

  @override
  Future<void> addChunks(List<DocumentChunk> chunks) async {
    _chunks.addAll(chunks);
    await _autoSave();
  }

  /// Searches for [topK] chunks most similar to [queryEmbedding].
  ///
  /// Algorithm: cosine similarity via [VectorSimilarity.cosine].
  /// Complexity: O(n * d), where n = number of chunks, d = dimensionality.
  @override
  Future<List<VectorSearchResult>> search(
    List<double> queryEmbedding, {
    int topK = 5,
    double minSimilarity = 0.0,
  }) async {
    final results = <VectorSearchResult>[];

    for (final chunk in _chunks) {
      if (chunk.embedding == null) continue;

      final similarity = VectorSimilarity.cosine(
        queryEmbedding,
        chunk.embedding!,
      );
      if (similarity >= minSimilarity) {
        results.add(
          VectorSearchResult(
            chunk: chunk,
            similarity: similarity,
            rank: 0, // filled in below
          ),
        );
      }
    }

    // Sort descending by similarity
    results.sort((a, b) => b.similarity.compareTo(a.similarity));

    // Assign rankings and limit to topK
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
    // Delete the JSON file if it exists
    if (autoSavePath != null) {
      final file = File(autoSavePath!);
      if (file.existsSync()) {
        await file.delete();
      }
    }
  }

  // ── Persistence ────────────────────────────────────────────────────────

  /// Loads the index from a JSON file and sets [autoSavePath].
  ///
  /// If the file does not exist — the store remains empty (no exception thrown).
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
      // Corrupted file — start with an empty store, do not interrupt execution
      _chunks.clear();
    }
  }

  // ── Getters ────────────────────────────────────────────────────────────

  @override
  int get size => _chunks.length;

  @override
  int get indexedSize => _chunks.where((c) => c.hasEmbedding).length;

  @override
  List<String> get documentIds =>
      _chunks.map((c) => c.documentId).toSet().toList();

  /// All chunks (including those without embeddings)
  List<DocumentChunk> get allChunks => List.unmodifiable(_chunks);

  // ── Private helpers ────────────────────────────────────────────────────

  /// Saves the current state to [autoSavePath] if it is set.
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
