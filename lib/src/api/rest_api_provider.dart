// lib/src/api/rest_api_provider.dart

/// Abstract interface for REST API LLM providers.
///
/// Implement this interface to add support for OpenAI, Anthropic, Google
/// Gemini, Ollama, or any other HTTP-based LLM API.
///
/// Typical lifecycle:
/// ```dart
/// final provider = OpenAIProvider();
/// await provider.initialize({'apiKey': 'sk-...', 'model': 'gpt-4o-mini'});
///
/// // Streaming
/// await for (final token in provider.sendPrompt('Hello')) {
///   print(token);
/// }
///
/// // Blocking
/// final response = await provider.sendPromptComplete('Hello');
///
/// await provider.dispose();
/// ```
abstract interface class RestApiProvider {
  /// Initializes the provider with the given configuration.
  ///
  /// Config keys are provider-specific. Throws [ArgumentError] if required
  /// keys are missing.
  Future<void> initialize(Map<String, dynamic> config);

  /// Sends a prompt and returns a stream of tokens in real time.
  ///
  /// Optional [parameters] are provider-specific (e.g. `temperature`,
  /// `maxTokens`, `systemPrompt`).
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  });

  /// Sends a prompt and returns the complete response as a [String].
  ///
  /// Blocks until the full response is received.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Future<String> sendPromptComplete(
    String prompt, {
    Map<String, dynamic>? parameters,
  });

  /// Whether the provider is initialized and ready for generation.
  bool get isInitialized;

  /// Releases all resources (HTTP connections, etc.).
  ///
  /// After calling this, the provider cannot be used again.
  Future<void> dispose();
}
