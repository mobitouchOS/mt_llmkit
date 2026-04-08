// lib/src/api/gemini_chat_provider.dart

import 'dart:async';
import 'dart:convert';
import 'dart:developer' as dev;

import 'package:http/http.dart' as http;

import 'ai_chat_provider.dart';
import 'chat_exceptions.dart';
import 'chat_models.dart';

/// [AIChatProvider] implementation for the Google Gemini API.
///
/// Uses the `generativelanguage.googleapis.com` REST endpoint.
///
/// Supported config keys:
/// - `apiKey` *(required)*: Google AI Studio API key
/// - `model` *(optional)*: model name (default: `"gemini-1.5-flash"`)
///
/// Role mapping:
/// - [ChatRole.user] → `"user"`
/// - [ChatRole.assistant] → `"model"` (Gemini convention)
/// - [ChatRole.system] messages are forwarded via the `system_instruction`
///   field, which is supported by Gemini 1.5+.
///
/// Example:
/// ```dart
/// final provider = GeminiChatProvider();
/// await provider.initialize({'apiKey': 'AIza...'});
///
/// final response = await provider.sendChatMessages([
///   ChatMessage.system('Reply only in haiku form.'),
///   ChatMessage.user('Explain recursion.'),
/// ]);
/// print(response.message.content);
///
/// await provider.dispose();
/// ```
class GeminiChatProvider extends BaseAIChatProvider {
  String _apiKey = '';
  String _model = 'gemini-1.5-flash';
  static const String _baseUrl =
      'https://generativelanguage.googleapis.com/v1beta/models';
  http.Client? _httpClient;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final apiKey = config['apiKey'] as String?;
    if (apiKey == null || apiKey.isEmpty) {
      throw ArgumentError.value(
        apiKey,
        'apiKey',
        'Gemini API key is required.',
      );
    }
    _apiKey = apiKey;
    _model = (config['model'] as String?) ?? _model;
    _httpClient = http.Client();
    markAsInitialized();
    dev.log('Initialized with model=$_model', name: 'GeminiChatProvider');
  }

  @override
  Future<ChatResponse> sendMessage(
    String message, {
    List<ChatMessage>? history,
    Map<String, dynamic>? parameters,
  }) {
    checkInitialized();
    return sendChatMessages([
      ...?history,
      ChatMessage.user(message),
    ], parameters: parameters);
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
    return _doStreamRequest([
      ...?history,
      ChatMessage.user(message),
    ], parameters);
  }

  @override
  Future<void> dispose() async {
    _httpClient?.close();
    _httpClient = null;
    await super.dispose();
    dev.log('Disposed.', name: 'GeminiChatProvider');
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  Future<ChatResponse> _doRequest(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters,
  ) async {
    final model = (parameters?['model'] as String?) ?? _model;
    final url = Uri.parse('$_baseUrl/$model:generateContent?key=$_apiKey');
    final body = _buildBody(messages, parameters);

    final http.Response response;
    try {
      response = await _httpClient!.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
    } catch (e) {
      throw NetworkException('HTTP request failed.', cause: e);
    }

    if (response.statusCode != 200) {
      dev.log(
        'Error ${response.statusCode}: ${response.body}',
        name: 'GeminiChatProvider',
      );
      throw mapHttpError(response.statusCode, response.body);
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return _parseResponse(json, model);
  }

  Stream<String> _doStreamRequest(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters,
  ) async* {
    final model = (parameters?['model'] as String?) ?? _model;
    // alt=sse enables Server-Sent Events format
    final url = Uri.parse(
      '$_baseUrl/$model:streamGenerateContent?key=$_apiKey&alt=sse',
    );
    final body = _buildBody(messages, parameters);
    final request = http.Request('POST', url)
      ..headers['Content-Type'] = 'application/json'
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
        name: 'GeminiChatProvider',
      );
      throw mapHttpError(response.statusCode, errorBody);
    }

    await for (final line
        in response.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
      if (line.isEmpty || !line.startsWith('data: ')) continue;
      try {
        final json = jsonDecode(line.substring(6)) as Map<String, dynamic>;
        final candidates = json['candidates'] as List?;
        if (candidates == null || candidates.isEmpty) continue;
        final content =
            (candidates.first as Map<String, dynamic>)['content']
                as Map<String, dynamic>?;
        final parts = content?['parts'] as List?;
        if (parts == null || parts.isEmpty) continue;
        final text = (parts.first as Map<String, dynamic>)['text'] as String?;
        if (text != null && text.isNotEmpty) yield text;
      } catch (e) {
        dev.log('Failed to parse SSE line: $line', name: 'GeminiChatProvider');
      }
    }
  }

  Map<String, dynamic> _buildBody(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters,
  ) {
    // Separate system messages — Gemini uses a dedicated field
    final systemMessages = messages
        .where((m) => m.role == ChatRole.system)
        .toList();
    final conversationMessages = messages
        .where((m) => m.role != ChatRole.system)
        .toList();

    final body = <String, dynamic>{
      'contents': conversationMessages
          .map(
            (m) => {
              'role': m.role == ChatRole.assistant ? 'model' : 'user',
              'parts': [
                {'text': m.content},
              ],
            },
          )
          .toList(),
    };

    if (systemMessages.isNotEmpty) {
      body['system_instruction'] = {
        'parts': [
          {'text': systemMessages.map((m) => m.content).join('\n')},
        ],
      };
    }

    final generationConfig = <String, dynamic>{};
    if (parameters?['temperature'] != null) {
      generationConfig['temperature'] = parameters!['temperature'];
    }
    if (parameters?['maxTokens'] != null) {
      generationConfig['maxOutputTokens'] = parameters!['maxTokens'];
    }
    if (generationConfig.isNotEmpty) {
      body['generationConfig'] = generationConfig;
    }

    return body;
  }

  ChatResponse _parseResponse(Map<String, dynamic> json, String model) {
    final candidate =
        (json['candidates'] as List).first as Map<String, dynamic>;
    final content = candidate['content'] as Map<String, dynamic>;
    final text = (content['parts'] as List).first['text'] as String;
    final usage = json['usageMetadata'] as Map<String, dynamic>?;
    return ChatResponse(
      message: ChatMessage(role: ChatRole.assistant, content: text),
      inputTokens: usage?['promptTokenCount'] as int?,
      outputTokens: usage?['candidatesTokenCount'] as int?,
      model: model,
      finishReason: candidate['finishReason'] as String?,
    );
  }
}
