// lib/src/rag/rag_pipeline.dart

import '../core/streaming_result.dart';
import '../presentation/llm_plugin.dart';
import 'chunking/text_chunker.dart';
import 'document/document.dart';
import 'embeddings/embedding_provider.dart';
import 'vector_store/vector_store.dart';

// ── Progress models ───────────────────────────────────────────────────────────

/// Postęp ingestii dokumentu (chunking + embedding).
class RagIngestionProgress {
  /// Całkowita liczba chunków do zaembeddowania
  final int totalChunks;

  /// Liczba chunków już zaembeddowanych
  final int embeddedChunks;

  /// Podgląd tekstu aktualnie przetwarzanego chunku (max 60 znaków)
  final String currentPreview;

  /// Czy ingestia dobiegła końca
  final bool isComplete;

  const RagIngestionProgress({
    required this.totalChunks,
    required this.embeddedChunks,
    required this.currentPreview,
    this.isComplete = false,
  });

  /// Ułamek ukończenia (0.0–1.0)
  double get fraction =>
      totalChunks > 0 ? embeddedChunks / totalChunks : 0.0;

  @override
  String toString() =>
      'RagIngestionProgress($embeddedChunks/$totalChunks, '
      '${(fraction * 100).toStringAsFixed(0)}%)';
}

// ── RagPipeline ───────────────────────────────────────────────────────────────

/// Orkiestrator pipeline'u RAG (Retrieval-Augmented Generation).
///
/// ## Wzorzec DI
///
/// Przyjmuje przez konstruktor:
/// - [EmbeddingProvider] — generowanie wektorów dla chunków i zapytań
/// - [VectorStore] — przechowywanie i wyszukiwanie wektorów
/// - [LlmPlugin] — generowanie odpowiedzi na podstawie kontekstu
/// - [TextChunker] — podział dokumentów na chunki (opcjonalny)
///
/// ## Pipeline ingestii
///
/// ```
/// Document → TextChunker → chunks → EmbeddingProvider.embed() → VectorStore.addChunks()
/// ```
///
/// ## Pipeline zapytania
///
/// ```
/// question → embed() → VectorStore.search() → build prompt → LlmPlugin.sendPromptStream()
/// ```
///
/// ## Przykład użycia
///
/// ```dart
/// final pipeline = RagPipeline(
///   embeddingProvider: LlamaEmbeddingProvider(),
///   vectorStore: InMemoryVectorStore(autoSavePath: '/path/to/index.json'),
///   generationPlugin: LlmPlugin.localGGUF(modelPath: '/path/to/llama.gguf'),
/// );
///
/// // Inicjalizacja
/// await pipeline.embeddingProvider.initialize({'modelPath': '/path/to/embed.gguf'});
/// await pipeline.generationPlugin.initialize();
///
/// // Ingestia dokumentu
/// final doc = Document.fromText(text, source: 'plik.txt');
/// await for (final progress in pipeline.ingestDocument(doc)) {
///   print('${progress.embeddedChunks}/${progress.totalChunks}');
/// }
///
/// // Zapytanie z RAG
/// await for (final chunk in pipeline.query('Co to jest RAG?')) {
///   print(chunk.text);
/// }
/// ```
class RagPipeline {
  final EmbeddingProvider embeddingProvider;
  final VectorStore vectorStore;
  final LlmPlugin generationPlugin;
  final TextChunker chunker;
  final String promptTemplate;

  /// Domyślny template promptu RAG z placeholderami `{context}` i `{question}`.
  ///
  /// Kontekst budowany jest z `topK` najlepiej pasujących chunków,
  /// oddzielonych `---`.
  static const String defaultPromptTemplate =
      'Na podstawie poniższego kontekstu odpowiedz na pytanie. '
      'Jeśli odpowiedź nie wynika z kontekstu, powiedz o tym wprost — nie wymyślaj informacji.\n'
      '\nKONTEKST:\n{context}\n\nPYTANIE: {question}\n\nODPOWIEDZ:';

  RagPipeline({
    required this.embeddingProvider,
    required this.vectorStore,
    required this.generationPlugin,
    TextChunker? chunker,
    String? promptTemplate,
  })  : chunker = chunker ?? const TextChunker(),
        promptTemplate = promptTemplate ?? defaultPromptTemplate;

  // ── Ingestia ──────────────────────────────────────────────────────────────

  /// Dzieli [document] na chunki, generuje embeddingi i zapisuje do [vectorStore].
  ///
  /// Zwraca stream [RagIngestionProgress] — można śledzić postęp w UI:
  ///
  /// ```dart
  /// await for (final p in pipeline.ingestDocument(doc)) {
  ///   setState(() => progress = p.fraction);
  /// }
  /// ```
  ///
  /// Po zakończeniu ostatni event ma [RagIngestionProgress.isComplete] = true.
  Stream<RagIngestionProgress> ingestDocument(Document document) async* {
    if (!embeddingProvider.isInitialized) {
      throw StateError(
        'EmbeddingProvider nie jest zainicjalizowany. '
        'Wywołaj embeddingProvider.initialize() przed ingestią.',
      );
    }

    // Krok 1: podział na chunki
    final chunks = chunker.chunk(document);
    if (chunks.isEmpty) return;

    yield RagIngestionProgress(
      totalChunks: chunks.length,
      embeddedChunks: 0,
      currentPreview: 'Dzielenie na ${chunks.length} chunków...',
    );

    // Krok 2: embedding każdego chunku
    for (int i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      chunk.embedding = await embeddingProvider.embed(chunk.text);

      yield RagIngestionProgress(
        totalChunks: chunks.length,
        embeddedChunks: i + 1,
        currentPreview: chunk.text.length > 60
            ? chunk.text.substring(0, 60)
            : chunk.text,
      );
    }

    // Krok 3: zapis do VectorStore (z automatycznym persist jeśli skonfigurowane)
    await vectorStore.addChunks(chunks);

    yield RagIngestionProgress(
      totalChunks: chunks.length,
      embeddedChunks: chunks.length,
      currentPreview: 'Zapisano ${chunks.length} chunków do indeksu.',
      isComplete: true,
    );
  }

  // ── Zapytanie ─────────────────────────────────────────────────────────────

  /// Wyszukuje relevantne chunki i generuje odpowiedź przez [generationPlugin].
  ///
  /// Zwraca [Stream<StreamingChunk>] — tokeny odpowiedzi pojawiają się na bieżąco.
  ///
  /// [topK] — liczba chunków kontekstu (domyślnie 5)
  /// [minSimilarity] — minimalne podobieństwo do włączenia w kontekst (0.0–1.0)
  ///
  /// Rzuca [StateError] jeśli żaden dokument nie jest zaindeksowany.
  Stream<StreamingChunk> query(
    String question, {
    int topK = 5,
    double minSimilarity = 0.25,
  }) async* {
    if (!embeddingProvider.isInitialized) {
      throw StateError('EmbeddingProvider nie jest zainicjalizowany.');
    }
    if (!generationPlugin.isInitialized) {
      throw StateError('GenerationPlugin nie jest zainicjalizowany.');
    }
    if (vectorStore.indexedSize == 0) {
      throw StateError(
        'Baza wektorów jest pusta. Dodaj dokumenty przez ingestDocument().',
      );
    }

    // Krok 1: embed zapytania
    final queryEmbedding = await embeddingProvider.embed(question);

    // Krok 2: znajdź relevantne chunki
    final results = await vectorStore.search(
      queryEmbedding,
      topK: topK,
      minSimilarity: minSimilarity,
    );

    if (results.isEmpty) {
      // Fallback: odpowiedź bez kontekstu z informacją dla użytkownika
      yield* generationPlugin.sendPromptStream(
        'Nie znaleziono relevantnych informacji w bazie wiedzy dla pytania: "$question". '
        'Poinformuj użytkownika, że baza nie zawiera odpowiednich danych.',
      );
      return;
    }

    // Krok 3: buduj prompt z kontekstem
    final contextParts = results.map((r) {
      final source = r.chunk.metadata['documentTitle'] ?? r.chunk.documentId;
      return '[$source, podobieństwo: ${(r.similarity * 100).toStringAsFixed(0)}%]\n${r.chunk.text}';
    }).join('\n\n---\n\n');

    final augmentedPrompt = promptTemplate
        .replaceAll('{context}', contextParts)
        .replaceAll('{question}', question);

    // Krok 4: generuj streaming odpowiedź
    yield* generationPlugin.sendPromptStream(augmentedPrompt);
  }

  /// Wyszukuje relevantne chunki dla [question] bez generowania odpowiedzi.
  ///
  /// Przydatne do inspekcji — sprawdzenia co RAG znalazł w bazie.
  Future<List<VectorSearchResult>> findRelevant(
    String question, {
    int topK = 5,
    double minSimilarity = 0.0,
  }) async {
    if (!embeddingProvider.isInitialized) {
      throw StateError('EmbeddingProvider nie jest zainicjalizowany.');
    }
    final queryEmbedding = await embeddingProvider.embed(question);
    return vectorStore.search(
      queryEmbedding,
      topK: topK,
      minSimilarity: minSimilarity,
    );
  }

  // ── Persistence ───────────────────────────────────────────────────────────

  /// Ładuje indeks wektorów z pliku JSON i ustawia ścieżkę auto-save.
  Future<void> loadIndex(String path) => vectorStore.load(path);

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Zwalnia zasoby embedding providera i modelu generowania.
  ///
  /// Nie zwalnia VectorStore — dane w pamięci są zachowane do końca sesji
  /// (i persisted na dysk jeśli autoSavePath był ustawiony).
  Future<void> dispose() async {
    await embeddingProvider.dispose();
    await generationPlugin.dispose();
  }
}
