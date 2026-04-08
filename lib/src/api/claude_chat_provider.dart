// lib/src/api/claude_chat_provider.dart

import 'dart:async';
import 'dart:convert';
import 'dart:developer' as dev;

import 'package:http/http.dart' as http;

import 'ai_chat_provider.dart';
import 'chat_exceptions.dart';
import 'chat_models.dart';

/// [AIChatProvider] implementation for the Anthropic Claude Messages API.
///
/// Supported config keys:
/// - `apiKey` *(required)*: Anthropic API key (format: `"sk-ant-..."`)
/// - `model` *(optional)*: model name
///   (default: `"claude-haiku-4-5-20251001"`)
/// - `anthropicVersion` *(optional)*: API version header
///   (default: `"2023-06-01"`)
///
/// [ChatRole.system] messages are extracted from the conversation list and
/// forwarded as the top-level `"system"` field required by the Anthropic
/// API.  Multiple system messages are joined with newlines.
///
/// Example:
/// ```dart
/// final provider = ClaudeChatProvider();
/// await provider.initialize({'apiKey': 'sk-ant-...'});
///
/// final response = await provider.sendChatMessages([
///   ChatMessage.system('You are a Dart expert.'),
///   ChatMessage.user('What is a mixin?'),
/// ]);
/// print(response.message.content);
///
/// await provider.dispose();
/// ```
class ClaudeChatProvider extends BaseAIChatProvider {
  String _apiKey = '';
  String _model = 'claude-haiku-4-5-20251001';
  String _anthropicVersion = '2023-06-01';
  static const String _baseUrl = 'https://api.anthropic.com/v1';
  http.Client? _httpClient;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    final apiKey = config['apiKey'] as String?;
    if (apiKey == null || apiKey.isEmpty) {
      throw ArgumentError.value(
        apiKey,
        'apiKey',
        'Anthropic API key is required.',
      );
    }
    _apiKey = apiKey;
    _model = (config['model'] as String?) ?? _model;
    _anthropicVersion =
        (config['anthropicVersion'] as String?) ?? _anthropicVersion;
    _httpClient = http.Client();
    markAsInitialized();
    dev.log('Initialized with model=$_model', name: 'ClaudeChatProvider');
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
    dev.log('Disposed.', name: 'ClaudeChatProvider');
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
        Uri.parse('$_baseUrl/messages'),
        headers: _headers(),
        body: jsonEncode(body),
      );
    } catch (e) {
      throw NetworkException('HTTP request failed.', cause: e);
    }

    if (response.statusCode != 200) {
      dev.log(
        'Error ${response.statusCode}: ${response.body}',
        name: 'ClaudeChatProvider',
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
    final request = http.Request('POST', Uri.parse('$_baseUrl/messages'))
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
        name: 'ClaudeChatProvider',
      );
      throw mapHttpError(response.statusCode, errorBody);
    }

    // Claude SSE pairs "event: <type>" lines with "data: {json}" lines.
    // We only care about `content_block_delta` events.
    String? currentEvent;
    await for (final line
        in response.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
      if (line.startsWith('event: ')) {
        currentEvent = line.substring(7).trim();
        continue;
      }
      if (!line.startsWith('data: ')) continue;

      if (currentEvent == 'content_block_delta') {
        try {
          final json = jsonDecode(line.substring(6)) as Map<String, dynamic>;
          final delta = json['delta'] as Map<String, dynamic>?;
          if (delta?['type'] == 'text_delta') {
            final text = delta!['text'] as String?;
            if (text != null && text.isNotEmpty) yield text;
          }
        } catch (e) {
          dev.log(
            'Failed to parse SSE line: $line',
            name: 'ClaudeChatProvider',
          );
        }
      }
    }
  }

  Map<String, dynamic> _buildBody(
    List<ChatMessage> messages,
    Map<String, dynamic>? parameters, {
    required bool stream,
  }) {
    // Claude requires system messages as a separate top-level field
    final systemMessages = messages
        .where((m) => m.role == ChatRole.system)
        .toList();
    final conversationMessages = messages
        .where((m) => m.role != ChatRole.system)
        .toList();

    final body = <String, dynamic>{
      'model': (parameters?['model'] as String?) ?? _model,
      'max_tokens': (parameters?['maxTokens'] as int?) ?? 1024,
      'messages': conversationMessages
          .map((m) => {'role': m.role.name, 'content': m.content})
          .toList(),
    };

    if (systemMessages.isNotEmpty) {
      body['system'] = systemMessages.map((m) => m.content).join('\n');
    }
    if (parameters?['temperature'] != null) {
      body['temperature'] = parameters!['temperature'];
    }
    if (stream) body['stream'] = true;

    return body;
  }

  Map<String, String> _headers() => {
    'x-api-key': _apiKey,
    'anthropic-version': _anthropicVersion,
    'Content-Type': 'application/json',
  };

  ChatResponse _parseResponse(Map<String, dynamic> json) {
    final contentBlocks = json['content'] as List;
    // Concatenate all text blocks (in practice usually just one)
    final text = contentBlocks
        .where((b) => (b as Map<String, dynamic>)['type'] == 'text')
        .map((b) => (b as Map<String, dynamic>)['text'] as String)
        .join();
    final usage = json['usage'] as Map<String, dynamic>?;
    return ChatResponse(
      message: ChatMessage(role: ChatRole.assistant, content: text),
      inputTokens: usage?['input_tokens'] as int?,
      outputTokens: usage?['output_tokens'] as int?,
      model: json['model'] as String?,
      finishReason: json['stop_reason'] as String?,
    );
  }
}
