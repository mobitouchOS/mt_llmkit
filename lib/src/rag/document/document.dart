// lib/src/rag/document/document.dart

import 'dart:convert';

/// Represents an indexed document in the RAG knowledge base.
///
/// A document is raw text extracted from a file (PDF, TXT, etc.)
/// together with metadata. Before use in RAG it must be split
/// into chunks ([TextChunker]) and embedded ([EmbeddingProvider]).
class Document {
  /// Unique document identifier (generated from source + timestamp)
  final String id;

  /// Path to the source file or a label (e.g. "manual", "url:...")
  final String source;

  /// Full text content of the document
  final String content;

  /// Document metadata.
  /// Typical keys: 'title', 'type' ('pdf'/'txt'), 'pages', 'sizeBytes', 'author'
  final Map<String, dynamic> metadata;

  /// Date the document was added to the store
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

  /// Creates a document from a text file.
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

  /// Creates a document from extracted PDF text.
  ///
  /// [source] — path to the PDF file
  /// [content] — text extracted by a PDF parser (e.g. syncfusion_flutter_pdf)
  /// [pageCount] — number of pages in the PDF
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

  // ── Helper getters ───────────────────────────────────────────────────────

  /// Number of characters in the content
  int get contentLength => content.length;

  /// Estimated word count
  int get wordCount => content.split(RegExp(r'\s+')).length;

  /// Document type ('pdf', 'txt', or 'unknown')
  String get type => metadata['type'] as String? ?? 'unknown';

  /// Document title (filename or from metadata)
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

  // ── Private helpers ──────────────────────────────────────────────────────

  static String _generateId(String source) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final hash = source.hashCode.abs();
    return 'doc_${hash}_$timestamp';
  }

  static String _filenameFromPath(String path) {
    if (path.isEmpty) return 'Document';
    final parts = path.replaceAll('\\', '/').split('/');
    final filename = parts.last;
    // Strip extension
    final dotIndex = filename.lastIndexOf('.');
    return dotIndex > 0 ? filename.substring(0, dotIndex) : filename;
  }
}
