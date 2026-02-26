// lib/src/api/ai_chat_provider.dart

import 'dart:developer' as dev;

import 'chat_exceptions.dart';
import 'chat_models.dart';

/// Abstract interface for AI chat providers.
///
/// Concrete implementations exist for OpenAI, Google Gemini, Anthropic
/// Claude, and Mistral AI.  All providers share the same lifecycle:
///
/// ```dart
/// final provider = OpenAIChatProvider();
/// await provider.initialize({'apiKey': 'sk-...'});
///
/// // Single-turn message
/// final response = await provider.sendMessage('What is Flutter?');
/// print(response.message.content);
///
/// // Multi-turn conversation
/// final history = [
///   ChatMessage.system('Reply only in Polish.'),
///   ChatMessage.user('Hello'),
///   ChatMessage.assistant('Cześć!'),
/// ];
/// final r2 = await provider.sendMessage('How are you?', history: history);
///
/// // Token streaming
/// await for (final token in provider.sendMessageStream('Tell me a joke')) {
///   stdout.write(token);
/// }
///
/// await provider.dispose();
/// ```
abstract interface class AIChatProvider {
  /// Initialises the provider with provider-specific [config].
  ///
  /// Required and optional config keys are documented per implementation.
  /// Throws [ArgumentError] if required keys are absent.
  Future<void> initialize(Map<String, dynamic> config);

  /// Sends a single [message] and returns the model's reply.
  ///
  /// Pass [history] to give the model prior conversation context.
  /// Throws [StateError] if the provider is not initialised.
  Future<ChatResponse> sendMessage(
    String message, {
    List<ChatMessage>? history,
    Map<String, dynamic>? parameters,
  });

  /// Sends a complete list of [messages] and returns the model's reply.
  ///
  /// Gives fine-grained control over conversation history including system
  /// instructions.
  /// Throws [StateError] if the provider is not initialised.
  Future<ChatResponse> sendChatMessages(
    List<ChatMessage> messages, {
    Map<String, dynamic>? parameters,
  });

  /// Sends a [message] and streams response tokens as they are generated.
  ///
  /// Pass [history] to give the model prior conversation context.
  /// Throws [StateError] if the provider is not initialised.
  Stream<String> sendMessageStream(
    String message, {
    List<ChatMessage>? history,
    Map<String, dynamic>? parameters,
  });

  /// Whether the provider has been initialised and is ready for use.
  bool get isInitialized;

  /// Releases all resources held by this provider (HTTP connections, etc.).
  ///
  /// After calling [dispose] the provider cannot be used again.
  Future<void> dispose();
}

// ── Base class ──────────────────────────────────────────────────────────────

/// Abstract base class that provides shared state management and utilities
/// for all [AIChatProvider] implementations.
///
/// Concrete providers extend this class and rely on:
/// - [markAsInitialized] / [markAsDisposed] for lifecycle state
/// - [checkInitialized] to guard public methods
/// - [withRetry] for automatic exponential-backoff retries
/// - [mapHttpError] to convert HTTP errors into typed exceptions
abstract class BaseAIChatProvider implements AIChatProvider {
  bool _isInitialized = false;
  bool _isDisposed = false;

  @override
  bool get isInitialized => _isInitialized;

  /// Marks the provider as ready.  Call at the end of [initialize].
  void markAsInitialized() => _isInitialized = true;

  /// Marks the provider as disposed.  Call at the start of [dispose].
  void markAsDisposed() {
    _isInitialized = false;
    _isDisposed = true;
  }

  /// Throws [StateError] if the provider has been disposed or not yet
  /// initialised.
  void checkInitialized() {
    if (_isDisposed) {
      throw StateError('Provider has been disposed.');
    }
    if (!_isInitialized) {
      throw StateError(
        'Provider is not initialised. Call initialize() first.',
      );
    }
  }

  /// Retries [fn] up to [maxAttempts] times with exponential back-off.
  ///
  /// Only [NetworkException] and [RateLimitException] trigger a retry;
  /// all other exceptions propagate immediately without retry.
  Future<T> withRetry<T>(
    Future<T> Function() fn, {
    int maxAttempts = 3,
  }) async {
    int attempt = 0;
    Duration delay = const Duration(seconds: 1);
    while (true) {
      try {
        attempt++;
        return await fn();
      } on RateLimitException catch (e) {
        if (attempt >= maxAttempts) rethrow;
        final wait = e.retryAfter ?? delay;
        dev.log(
          'Rate limit hit. Retrying in ${wait.inSeconds}s '
          '(attempt $attempt/$maxAttempts).',
          name: 'AIChatProvider',
        );
        await Future<void>.delayed(wait);
        delay = delay * 2;
      } on NetworkException catch (e) {
        if (attempt >= maxAttempts) rethrow;
        dev.log(
          'Network error: ${e.message}. Retrying in ${delay.inSeconds}s '
          '(attempt $attempt/$maxAttempts).',
          name: 'AIChatProvider',
        );
        await Future<void>.delayed(delay);
        delay = delay * 2;
      }
    }
  }

  /// Converts an HTTP [statusCode] and response [body] into the appropriate
  /// [AIChatException] subclass.
  ///
  /// | Status | Exception |
  /// |--------|-----------|
  /// | 401, 403 | [APIKeyException] |
  /// | 429 | [RateLimitException] |
  /// | other | [AIChatException] |
  AIChatException mapHttpError(int statusCode, String body) {
    return switch (statusCode) {
      401 || 403 => APIKeyException(
          'Authentication failed: $body',
          statusCode: statusCode,
        ),
      429 => RateLimitException(
          'Rate limit exceeded: $body',
          retryAfter: _parseRetryAfter(body),
        ),
      _ => AIChatException('API error: $body', statusCode: statusCode),
    };
  }

  /// Attempts to extract a retry delay embedded in the response [body].
  Duration? _parseRetryAfter(String body) {
    final match = RegExp(r'"retry_after"\s*:\s*(\d+)').firstMatch(body);
    if (match != null) {
      final seconds = int.tryParse(match.group(1)!);
      if (seconds != null) return Duration(seconds: seconds);
    }
    return null;
  }

  @override
  Future<void> dispose() async {
    markAsDisposed();
  }
}
