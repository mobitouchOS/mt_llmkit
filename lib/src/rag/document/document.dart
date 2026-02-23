// lib/src/rag/document/document.dart

import 'dart:convert';

/// Reprezentuje zaindeksowany dokument w bazie wiedzy RAG.
///
/// Dokument to surowy tekst wyekstrahowany z pliku (PDF, TXT itp.)
/// wraz z metadanymi. Przed użyciem w RAG musi zostać podzielony
/// na chunki ([TextChunker]) i osadzony ([EmbeddingProvider]).
class Document {
  /// Unikalny identyfikator dokumentu (generowany na podstawie source + timestamp)
  final String id;

  /// Ścieżka do pliku źródłowego lub etykieta (np. "manual", "url:...")
  final String source;

  /// Pełna treść tekstowa dokumentu
  final String content;

  /// Metadane dokumentu
  /// Typowe klucze: 'title', 'type' ('pdf'/'txt'), 'pages', 'sizeBytes', 'author'
  final Map<String, dynamic> metadata;

  /// Data dodania dokumentu do bazy
  final DateTime createdAt;

  Document({
    required this.id,
    required this.source,
    required this.content,
    Map<String, dynamic>? metadata,
    DateTime? createdAt,
  })  : metadata = metadata ?? {},
        createdAt = createdAt ?? DateTime.now();

  // ── Factory constructors ─────────────────────────────────────────────────

  /// Tworzy dokument z pliku tekstowego.
  factory Document.fromText(
    String content, {
    required String source,
    Map<String, dynamic>? metadata,
  }) {
    final id = _generateId(source);
    return Document(
      id: id,
      source: source,
      content: content,
      metadata: {
        'type': 'txt',
        'title': _filenameFromPath(source),
        'sizeBytes': utf8.encode(content).length,
        ...?metadata,
      },
    );
  }

  /// Tworzy dokument z wyekstrahowanego tekstu PDF.
  ///
  /// [source] — ścieżka do pliku PDF
  /// [content] — tekst wyekstrahowany przez parser PDF (np. syncfusion_flutter_pdf)
  /// [pageCount] — liczba stron w PDF
  factory Document.fromPdf(
    String content, {
    required String source,
    int? pageCount,
    Map<String, dynamic>? metadata,
  }) {
    final id = _generateId(source);
    return Document(
      id: id,
      source: source,
      content: content,
      metadata: {
        'type': 'pdf',
        'title': _filenameFromPath(source),
        'pages': pageCount,
        'sizeBytes': utf8.encode(content).length,
        ...?metadata,
      },
    );
  }

  // ── Gettery pomocnicze ──────────────────────────────────────────────────

  /// Liczba znaków w treści
  int get contentLength => content.length;

  /// Szacowana liczba słów
  int get wordCount => content.split(RegExp(r'\s+')).length;

  /// Typ dokumentu ('pdf', 'txt', lub 'unknown')
  String get type => metadata['type'] as String? ?? 'unknown';

  /// Tytuł dokumentu (nazwa pliku lub z metadanych)
  String get title => metadata['title'] as String? ?? source;

  // ── Serialization ────────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'id': id,
        'source': source,
        'content': content,
        'metadata': metadata,
        'createdAt': createdAt.toIso8601String(),
      };

  factory Document.fromJson(Map<String, dynamic> json) => Document(
        id: json['id'] as String,
        source: json['source'] as String,
        content: json['content'] as String,
        metadata: (json['metadata'] as Map<String, dynamic>?) ?? {},
        createdAt: DateTime.parse(json['createdAt'] as String),
      );

  @override
  String toString() =>
      'Document(id: $id, title: $title, chars: $contentLength)';

  // ── Prywatne helpery ─────────────────────────────────────────────────────

  static String _generateId(String source) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final hash = source.hashCode.abs();
    return 'doc_${hash}_$timestamp';
  }

  static String _filenameFromPath(String path) {
    if (path.isEmpty) return 'Dokument';
    final parts = path.replaceAll('\\', '/').split('/');
    final filename = parts.last;
    // Usuń rozszerzenie
    final dotIndex = filename.lastIndexOf('.');
    return dotIndex > 0 ? filename.substring(0, dotIndex) : filename;
  }
}
