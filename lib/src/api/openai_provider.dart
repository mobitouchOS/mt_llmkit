// lib/src/api/openai_provider.dart

import 'rest_api_provider.dart';

/// Skeleton implementation of [RestApiProvider] for the OpenAI API.
///
/// ## Status: Not implemented — ready for integration
///
/// Methods throw [UnimplementedError] with TODO instructions.
///
/// ## Requirements for full implementation
///
/// Add dependencies in `pubspec.yaml`:
/// ```yaml
/// dependencies:
///   http: ^1.2.0          # or dio: ^5.7.0
///   # alternatively the official client:
///   # openai_dart: ^0.4.0
/// ```
///
/// ## Target data flow architecture
///
/// ```
/// UI Thread
///    │  listen()
///    ▼
/// StreamController<String>   ← safe boundary
///    ▲
///    │  add(token)
/// SSE Parser                 ← Server-Sent Events parsing
///    ▲
///    │  HTTP chunked response
/// OpenAI API                 ← /v1/chat/completions?stream=true
/// ```
class OpenAIProvider implements RestApiProvider {
  String? _apiKey;
  final String _model = 'gpt-4o-mini';
  final String _baseUrl = 'https://api.openai.com/v1';
  bool _isInitialized = false;

  // TODO: Initialize HTTP client
  // final http.Client _httpClient = http.Client();

  /// Initializes the OpenAI provider with an API key.
  ///
  /// Required keys in [config]:
  ///   - `apiKey` (String): OpenAI API key (format: "sk-...")
  ///
  /// Optional keys:
  ///   - `model` (String): model name (default: "gpt-4o-mini")
  ///   - `baseUrl` (String): base API URL (default: OpenAI, can be changed
  ///     e.g. to Azure)
  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    // TODO: Implement OpenAI provider initialization
    throw UnimplementedError(
      'OpenAI provider is not yet implemented.\n'
      'TODO: Add the "http" package and implement the initialize() method.',
    );
  }

  /// Sends a prompt to the OpenAI API and returns a token stream via SSE.
  @override
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    // TODO: Implement SSE streaming from the OpenAI API
    throw UnimplementedError(
      'OpenAI streaming is not yet implemented.\n'
      'TODO: Implement SSE parsing from /v1/chat/completions.',
    );
  }

  /// Sends a prompt and returns the complete response as a [String].
  @override
  Future<String> sendPromptComplete(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    // TODO: Implement non-streaming request to the OpenAI API
    throw UnimplementedError(
      'OpenAI sendPromptComplete is not yet implemented.\n'
      'TODO: Implement POST /v1/chat/completions without streaming.',
    );
  }

  @override
  bool get isInitialized => _isInitialized;

  @override
  Future<void> dispose() async {
    // TODO: Close the HTTP client and cancel pending requests
    // _httpClient.close();
    _apiKey = null;
    _isInitialized = false;
  }

  // ── Diagnostic getters ─────────────────────────────────────────────────

  /// Name of the currently configured model.
  String get currentModel => _model;

  /// Base API URL.
  String get baseUrl => _baseUrl;

  /// Whether the provider has an API key (does not validate it).
  bool get hasApiKey => _apiKey != null && _apiKey!.isNotEmpty;
}
