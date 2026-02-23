// lib/src/rag/document/document_chunk.dart

/// Fragment dokumentu (chunk) z opcjonalnym wektorem embeddingu.
///
/// Chunki tworzone są przez [TextChunker] z [Document.content].
/// Pole [embedding] jest `null` dopóki nie zostanie uzupełnione
/// przez [EmbeddingProvider.embed(chunk.text)].
///
/// Serializacja JSON obsługuje `List<double>?` embedding.
class DocumentChunk {
  /// Unikalny identyfikator chunku: "${documentId}_chunk_${chunkIndex}"
  final String id;

  /// Identyfikator dokumentu źródłowego
  final String documentId;

  /// Tekst tego fragmentu
  final String text;

  /// Indeks chunku w dokumencie (0-based)
  final int chunkIndex;

  /// Pozycja startowa w oryginalnym tekście dokumentu (włącznie)
  final int startChar;

  /// Pozycja końcowa w oryginalnym tekście dokumentu (wyłącznie)
  final int endChar;

  /// Wektor embeddingu — `null` do czasu wywołania [EmbeddingProvider.embed].
  ///
  /// Mutable: ustawiany po wygenerowaniu embeddingu przez [RagPipeline].
  List<double>? embedding;

  /// Metadane chunku (dziedziczone z dokumentu lub dodatkowe)
  final Map<String, dynamic> metadata;

  DocumentChunk({
    required this.id,
    required this.documentId,
    required this.text,
    required this.chunkIndex,
    required this.startChar,
    required this.endChar,
    this.embedding,
    Map<String, dynamic>? metadata,
  }) : metadata = metadata ?? {};

  // ── Gettery ──────────────────────────────────────────────────────────────

  /// Czy ten chunk ma wygenerowany embedding
  bool get hasEmbedding => embedding != null;

  /// Liczba znaków w tekście chunka
  int get length => text.length;

  /// Podgląd tekstu (pierwsze 80 znaków)
  String get preview {
    if (text.length <= 80) return text;
    return '${text.substring(0, 77)}...';
  }

  // ── Serialization ────────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'id': id,
        'documentId': documentId,
        'text': text,
        'chunkIndex': chunkIndex,
        'startChar': startChar,
        'endChar': endChar,
        // List<double> → List<dynamic> (JSON compatible)
        'embedding': embedding,
        'metadata': metadata,
      };

  factory DocumentChunk.fromJson(Map<String, dynamic> json) {
    final rawEmbedding = json['embedding'] as List<dynamic>?;
    return DocumentChunk(
      id: json['id'] as String,
      documentId: json['documentId'] as String,
      text: json['text'] as String,
      chunkIndex: json['chunkIndex'] as int,
      startChar: json['startChar'] as int,
      endChar: json['endChar'] as int,
      embedding: rawEmbedding?.map((e) => (e as num).toDouble()).toList(),
      metadata: (json['metadata'] as Map<String, dynamic>?) ?? {},
    );
  }

  @override
  String toString() =>
      'DocumentChunk(id: $id, chars: $length, hasEmbedding: $hasEmbedding)';
}
