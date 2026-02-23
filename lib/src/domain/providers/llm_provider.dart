// lib/src/domain/providers/llm_provider.dart

/// Abstrakcyjny interfejs dla wszystkich dostawców LLM.
///
/// Wzorzec Strategy + Dependency Injection — każdy dostawca
/// (lokalny GGUF, OpenAI, Anthropic, Google itp.) implementuje ten interfejs,
/// co umożliwia podmianę dostawcy bez modyfikacji kodu klienta.
///
/// Typowy cykl życia:
/// ```dart
/// final provider = LocalGGUFProvider();
/// await provider.initialize({'modelPath': '/path/to/model.gguf'});
/// await for (final token in provider.sendPrompt('Hello')) {
///   print(token);
/// }
/// await provider.dispose();
/// ```
abstract interface class LLMProvider {
  /// Inicjalizuje dostawcę z podaną konfiguracją.
  ///
  /// Klucze mapy [config] są specyficzne dla każdego dostawcy:
  ///
  /// **LocalGGUFProvider:**
  ///   - `modelPath` (String, wymagany): ścieżka do pliku .gguf
  ///   - `llmConfig` (LlmConfig, opcjonalny): parametry modelu (temp, nCtx itp.)
  ///
  /// **OpenAIProvider:**
  ///   - `apiKey` (String, wymagany): klucz API (zaczyna się od "sk-")
  ///   - `model` (String, opcjonalny): nazwa modelu (domyślnie "gpt-4o-mini")
  ///   - `baseUrl` (String, opcjonalny): URL bazowy (np. Azure OpenAI)
  ///
  /// Rzuca [ArgumentError] jeśli brakuje wymaganych kluczy.
  /// Rzuca [FileSystemException] jeśli plik modelu nie istnieje (GGUF).
  Future<void> initialize(Map<String, dynamic> config);

  /// Wysyła prompt i zwraca stream tokenów w czasie rzeczywistym.
  ///
  /// Stream jest bezpieczny dla wątku UI — tokeny są generowane
  /// asynchronicznie (Isolate lub HTTP) i dostarczane przez [StreamController].
  ///
  /// Parametry [parameters] opcjonalne, specyficzne dla dostawcy:
  ///   - `promptFormat` (PromptFormat): format promptu (dla GGUF)
  ///   - `temperature` (double): temperatura próbkowania
  ///   - `maxTokens` (int): maksymalna liczba tokenów do wygenerowania
  ///   - `systemPrompt` (String): systemowy prompt (dla OpenAI)
  ///
  /// Rzuca [StateError] jeśli provider nie jest zainicjalizowany.
  Stream<String> sendPrompt(String prompt, {Map<String, dynamic>? parameters});

  /// Zwalnia wszystkie zasoby (Isolate, połączenia HTTP, pamięć modelu).
  ///
  /// Po wywołaniu provider nie może być ponownie użyty.
  Future<void> dispose();
}
