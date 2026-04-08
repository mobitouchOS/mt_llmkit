// lib/src/api/chat_exceptions.dart

/// Base exception for all AI chat provider errors.
///
/// Subclasses provide finer-grained error categorisation:
/// - [APIKeyException]  — authentication / authorisation failures
/// - [NetworkException] — transport-level failures
/// - [RateLimitException] — quota / rate-limit (HTTP 429)
class AIChatException implements Exception {
  const AIChatException(this.message, {this.statusCode, this.cause});

  /// Human-readable description of the error.
  final String message;

  /// HTTP status code, if the error originated from an HTTP response.
  final int? statusCode;

  /// Underlying exception or error that triggered this one, if any.
  final Object? cause;

  @override
  String toString() {
    final buffer = StringBuffer('AIChatException: $message');
    if (statusCode != null) buffer.write(' (HTTP $statusCode)');
    if (cause != null) buffer.write('\nCaused by: $cause');
    return buffer.toString();
  }
}

/// Thrown when the API key is missing, invalid, or unauthorised.
///
/// Typically maps to HTTP 401 or 403 responses.
class APIKeyException extends AIChatException {
  const APIKeyException(super.message, {super.statusCode, super.cause});

  @override
  String toString() =>
      'APIKeyException: $message'
      '${statusCode != null ? ' (HTTP $statusCode)' : ''}';
}

/// Thrown for transport-level failures such as timeouts, DNS errors,
/// or connection resets.
class NetworkException extends AIChatException {
  const NetworkException(super.message, {super.statusCode, super.cause});

  @override
  String toString() =>
      'NetworkException: $message'
      '${cause != null ? '\nCaused by: $cause' : ''}';
}

/// Thrown when the provider rate-limit is exceeded (HTTP 429).
///
/// [retryAfter] carries the suggested delay before the next attempt,
/// parsed from the `Retry-After` header or response body when available.
class RateLimitException extends AIChatException {
  const RateLimitException(
    super.message, {
    this.retryAfter,
    super.statusCode = 429,
    super.cause,
  });

  /// Suggested delay before the next attempt, or `null` if not provided.
  final Duration? retryAfter;

  @override
  String toString() {
    final retry = retryAfter != null
        ? ' Retry after ${retryAfter!.inSeconds}s.'
        : '';
    return 'RateLimitException: $message$retry';
  }
}
