// lib/src/domain/providers/llm_provider.dart

/// Abstract interface for all LLM providers.
///
/// Strategy + Dependency Injection pattern — each provider
/// (local GGUF, OpenAI, Anthropic, Google, etc.) implements this interface,
/// allowing the provider to be swapped without modifying client code.
///
/// Typical lifecycle:
/// ```dart
/// final provider = LocalGGUFProvider();
/// await provider.initialize({'modelPath': '/path/to/model.gguf'});
/// await for (final token in provider.sendPrompt('Hello')) {
///   print(token);
/// }
/// await provider.dispose();
/// ```
abstract interface class LLMProvider {
  /// Initializes the provider with the given configuration.
  ///
  /// Keys in [config] are provider-specific:
  ///
  /// **LocalGGUFProvider:**
  ///   - `modelPath` (String, required): path to the .gguf file
  ///   - `llmConfig` (LlmConfig, optional): model parameters (temp, nCtx, etc.)
  ///
  /// **OpenAIProvider:**
  ///   - `apiKey` (String, required): API key (starts with "sk-")
  ///   - `model` (String, optional): model name (default: "gpt-4o-mini")
  ///   - `baseUrl` (String, optional): base URL (e.g. Azure OpenAI)
  ///
  /// Throws [ArgumentError] if required keys are missing.
  /// Throws [FileSystemException] if the model file does not exist (GGUF).
  Future<void> initialize(Map<String, dynamic> config);

  /// Sends a prompt and returns a stream of tokens in real time.
  ///
  /// The stream is UI-thread-safe — tokens are generated
  /// asynchronously (Isolate or HTTP) and delivered via [StreamController].
  ///
  /// Optional [parameters], provider-specific:
  ///   - `promptFormat` (PromptFormat): prompt format (for GGUF)
  ///   - `temperature` (double): sampling temperature
  ///   - `maxTokens` (int): maximum number of tokens to generate
  ///   - `systemPrompt` (String): system prompt (for OpenAI)
  ///
  /// Throws [StateError] if the provider is not initialized.
  Stream<String> sendPrompt(String prompt, {Map<String, dynamic>? parameters});

  /// Releases all resources (Isolate, HTTP connections, model memory).
  ///
  /// After calling this, the provider cannot be used again.
  Future<void> dispose();
}
