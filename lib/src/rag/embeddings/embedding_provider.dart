// lib/src/rag/embeddings/embedding_provider.dart

/// Abstract interface for text embedding providers.
///
/// An embedding is a vector representation of text (`List<double>`) in a
/// semantic space — texts with similar meaning have close vectors.
/// Used by [InMemoryVectorStore] for cosine-similarity search.
///
/// ## Implementations
///
/// - [LlamaEmbeddingProvider] — local embeddings via llama.cpp (GGUF model)
///
/// ## Usage
///
/// ```dart
/// final provider = LlamaEmbeddingProvider();
/// await provider.initialize({'modelPath': '/path/to/nomic-embed-text.gguf'});
///
/// final vector = await provider.embed('What is the capital of Poland?');
/// print('Dimensions: ${vector.length}');  // e.g. 768 for nomic-embed-text
///
/// await provider.dispose();
/// ```
abstract interface class EmbeddingProvider {
  /// Initializes the provider with the given configuration.
  ///
  /// Required/optional keys depend on the implementation.
  /// For [LlamaEmbeddingProvider]:
  ///   - `modelPath` (String, required): path to the embeddings model (.gguf)
  ///   - `llmConfig` (LlmConfig, optional): context parameters
  Future<void> initialize(Map<String, dynamic> config);

  /// Generates an embedding for a single text.
  ///
  /// Returns an L2-normalised vector as [List<double>].
  /// Throws [StateError] if the provider is not initialized.
  Future<List<double>> embed(String text);

  /// Generates embeddings for multiple texts sequentially.
  ///
  /// Default implementation calls [embed()] in a loop.
  /// Implementations may override this method for batch processing efficiency.
  Future<List<List<double>>> embedBatch(List<String> texts);

  /// Releases resources (model, Isolate).
  Future<void> dispose();

  /// Dimensionality of the embedding vector.
  ///
  /// Returns 0 before [initialize()] is called.
  /// Example values: 384 (bge-small), 768 (nomic-embed-text), 1536 (text-embedding-3-small).
  int get dimensions;

  /// Whether the provider is initialized and ready for use
  bool get isInitialized;
}
