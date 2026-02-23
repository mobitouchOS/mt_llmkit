// lib/src/rag/embeddings/embedding_provider.dart

/// Abstrakcyjny interfejs dla dostawców embeddingów tekstowych.
///
/// Embedding to wektorowa reprezentacja tekstu (`List<double>`) w przestrzeni
/// semantycznej — podobne znaczeniowo teksty mają bliskie wektory.
/// Używany przez [InMemoryVectorStore] do wyszukiwania przez podobieństwo kosinusowe.
///
/// ## Implementacje
///
/// - [LlamaEmbeddingProvider] — lokalne embeddingi przez llama.cpp (GGUF model)
///
/// ## Użycie
///
/// ```dart
/// final provider = LlamaEmbeddingProvider();
/// await provider.initialize({'modelPath': '/path/to/nomic-embed-text.gguf'});
///
/// final vector = await provider.embed('Jaka jest stolica Polski?');
/// print('Wymiarowość: ${vector.length}');  // np. 768 dla nomic-embed-text
///
/// await provider.dispose();
/// ```
abstract interface class EmbeddingProvider {
  /// Inicjalizuje dostawcę z konfiguracją.
  ///
  /// Wymagane/opcjonalne klucze zależą od implementacji.
  /// Dla [LlamaEmbeddingProvider]:
  ///   - `modelPath` (String, wymagany): ścieżka do modelu embeddingów (.gguf)
  ///   - `llmConfig` (LlmConfig, opcjonalny): parametry kontekstu
  Future<void> initialize(Map<String, dynamic> config);

  /// Generuje embedding dla pojedynczego tekstu.
  ///
  /// Zwraca znormalizowany wektor L2 jako [List<double>].
  /// Rzuca [StateError] jeśli provider nie jest zainicjalizowany.
  Future<List<double>> embed(String text);

  /// Generuje embeddingi dla wielu tekstów sekwencyjnie.
  ///
  /// Domyślna implementacja wywołuje [embed()] w pętli.
  /// Implementacje mogą nadpisać tę metodę dla wydajności batch processing.
  Future<List<List<double>>> embedBatch(List<String> texts);

  /// Zwalnia zasoby (model, Isolate).
  Future<void> dispose();

  /// Wymiarowość wektora embeddingu.
  ///
  /// Zwraca 0 przed wywołaniem [initialize()].
  /// Przykładowe wartości: 384 (bge-small), 768 (nomic-embed-text), 1536 (text-embedding-3-small).
  int get dimensions;

  /// Czy provider jest zainicjalizowany i gotowy do użycia
  bool get isInitialized;
}
