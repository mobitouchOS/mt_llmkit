// lib/src/api/openai_chat_provider.dart

import 'dart:async';
import 'dart:convert';
import 'dart:developer' as dev;

import 'package:http/http.dart' as http;

import 'ai_chat_provider.dart';
import 'chat_exceptions.dart';
import 'chat_models.dart';

/// [AIChatProvider] implementation for the OpenAI Chat Completions API.
///
/// Supported config keys:
/// - `apiKey` *(required)*: OpenAI API key (format: `"sk-..."`)
/// - `model` *(optional)*: model name (default: `"gpt-4o-mini"`)
/// - `baseUrl` *(optional)*: override the base URL, e.g. for Azure
///   OpenAI deployments (default: `"https://api.openai.com/v1"`)
///
/// Example — basic conversation:
/// ```dart
/// final provider = OpenAIChatProvider();
/// await provider.initialize({'apiKey': 'sk-...'});
///
/// final history = [
///   ChatMessage.system('You are a concise assistant.'),
/// ];
/// final r1 = await provider.sendMessage('What is Dart?', history: history);
/// history.add(r1.message);
///
/// final r2 = await provider.sendMessage('Give me an example.', history: history);
/// print(r2.message.content);
///
/// await provider.dispose();
/// ```
///
/// Example — streaming:
/// ```dart
/// await for (final token in provider.sendMessageStream('Tell me a joke')) {
///   stdout.write(token);
/// }
/// ```
class OpenAIChatProvider extends BaseAIChatProvider {
  String _apiKey = '';
  String _model = 'gpt-4o-mini';
  String _baseUrl = 'https://api.openai.com/v1';
  http.Client? _httpClient;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final apiKey = config['apiKey'] as String?;
    if (apiKey == null || apiKey.isEmpty) {
      throw ArgumentError.value(
        apiKey,
        'apiKey',
        'OpenAI API key is required.',
      );
    }
    _apiKey = apiKey;
    _model = (config['model'] as String?) ?? _model;
    _baseUrl = (config['baseUrl'] as String?) ?? _baseUrl;
    _httpClient = http.Client();
    markAsInitialized();
    dev.log('Initialized with model=$_model', name: 'OpenAIChatProvider');
  }

  @override
  Future<ChatResponse> sendMessage(
    String message, {
    List<ChatMessage>? history,
    Map<String, dynamic>? parameters,
  }) {
    checkInitialized();
    return sendChatMessages(
      [...?history, ChatMessage.user(message)],
      parameters: parameters,
    );
  }

  @override
  Future<ChatResponse> sendChatMessages(
    List<ChatMessage> messages, {
    Map<String, dynamic>? parameters,
  }) {
    checkInitialized();
    return withRetry(() => _doRequest(messages, parameters));
  }

  @override
  Stream<String> sendMessageStream(
    String message, {
    List<ChatMessage>? history,
    Map<String, dynamic>? parameters,
  }) {
    checkInitialized();
    return _doStreamRequest(
      [...?history, ChatMessage.user(message)],
      parameters,
    );
  }

  @override
  Future<void> dispose() async {
    _httpClient?.close();
    _httpClient = null;
    await super.dispose();
    dev.log('Disposed.', name: 'OpenAIChatProvider');
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  Future<ChatResponse> _doRequest(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters,
  ) async {
    final body = _buildBody(messages, parameters, stream: false);
    final http.Response response;
    try {
      response = await _httpClient!.post(
        Uri.parse('$_baseUrl/chat/completions'),
        headers: _headers(),
        body: jsonEncode(body),
      );
    } catch (e) {
      throw NetworkException('HTTP request failed.', cause: e);
    }

    if (response.statusCode != 200) {
      dev.log(
        'Error ${response.statusCode}: ${response.body}',
        name: 'OpenAIChatProvider',
      );
      throw mapHttpError(response.statusCode, response.body);
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return _parseResponse(json);
  }

  Stream<String> _doStreamRequest(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters,
  ) async* {
    final body = _buildBody(messages, parameters, stream: true);
    final request = http.Request('POST', Uri.parse('$_baseUrl/chat/completions'))
      ..headers.addAll(_headers())
      ..body = jsonEncode(body);

    final http.StreamedResponse response;
    try {
      response = await _httpClient!.send(request);
    } catch (e) {
      throw NetworkException('HTTP request failed.', cause: e);
    }

    if (response.statusCode != 200) {
      final errorBody = await response.stream.bytesToString();
      dev.log(
        'Stream error ${response.statusCode}: $errorBody',
        name: 'OpenAIChatProvider',
      );
      throw mapHttpError(response.statusCode, errorBody);
    }

    // Each SSE line: "data: {json}" or "data: [DONE]"
    await for (final line in response.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter())) {
      if (line.isEmpty || line == 'data: [DONE]') continue;
      if (!line.startsWith('data: ')) continue;
      try {
        final json = jsonDecode(line.substring(6)) as Map<String, dynamic>;
        final choices = json['choices'] as List?;
        if (choices == null || choices.isEmpty) continue;
        final delta =
            (choices.first as Map<String, dynamic>)['delta'] as Map<String, dynamic>?;
        final token = delta?['content'] as String?;
        if (token != null && token.isNotEmpty) yield token;
      } catch (e) {
        dev.log(
          'Failed to parse SSE line: $line',
          name: 'OpenAIChatProvider',
        );
      }
    }
  }

  Map<String, dynamic> _buildBody(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters, {
    required bool stream,
  }) {
    return {
      'model': (parameters?['model'] as String?) ?? _model,
      'messages': messages.map((m) => m.toJson()).toList(),
      if (parameters?['temperature'] != null)
        'temperature': parameters!['temperature'],
      if (parameters?['maxTokens'] != null)
        'max_tokens': parameters!['maxTokens'],
      if (stream) 'stream': true,
    };
  }

  Map<String, String> _headers() => {
        'Authorization': 'Bearer $_apiKey',
        'Content-Type': 'application/json',
      };

  ChatResponse _parseResponse(Map<String, dynamic> json) {
    final choice = (json['choices'] as List).first as Map<String, dynamic>;
    final msgJson = choice['message'] as Map<String, dynamic>;
    final usage = json['usage'] as Map<String, dynamic>?;
    return ChatResponse(
      message: ChatMessage(
        role: ChatRole.assistant,
        content: msgJson['content'] as String,
      ),
      inputTokens: usage?['prompt_tokens'] as int?,
      outputTokens: usage?['completion_tokens'] as int?,
      model: json['model'] as String?,
      finishReason: choice['finish_reason'] as String?,
    );
  }
}
