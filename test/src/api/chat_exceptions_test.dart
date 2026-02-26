// test/src/api/chat_exceptions_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/src/api/chat_exceptions.dart';

void main() {
  group('AIChatException', () {
    test('stores message, statusCode, and cause', () {
      final ex = AIChatException(
        'Something went wrong',
        statusCode: 500,
        cause: 'underlying error',
      );
      expect(ex.message, 'Something went wrong');
      expect(ex.statusCode, 500);
      expect(ex.cause, 'underlying error');
    });

    test('optional fields default to null', () {
      const ex = AIChatException('Oops');
      expect(ex.statusCode, isNull);
      expect(ex.cause, isNull);
    });

    test('toString() includes message and statusCode', () {
      final ex = AIChatException('Oops', statusCode: 500);
      expect(ex.toString(), contains('Oops'));
      expect(ex.toString(), contains('500'));
    });

    test('toString() includes cause when set', () {
      final ex = AIChatException('Error', cause: 'root cause');
      expect(ex.toString(), contains('root cause'));
    });

    test('implements Exception', () {
      expect(const AIChatException('test'), isA<Exception>());
    });
  });

  group('APIKeyException', () {
    test('is a subtype of AIChatException', () {
      expect(const APIKeyException('bad key'), isA<AIChatException>());
    });

    test('implements Exception', () {
      expect(const APIKeyException('test'), isA<Exception>());
    });

    test('toString() starts with APIKeyException prefix', () {
      final ex = APIKeyException('Invalid key', statusCode: 401);
      expect(ex.toString(), startsWith('APIKeyException'));
      expect(ex.toString(), contains('Invalid key'));
    });

    test('statusCode is stored correctly', () {
      final ex = APIKeyException('Auth failed', statusCode: 403);
      expect(ex.statusCode, 403);
    });
  });

  group('NetworkException', () {
    test('is a subtype of AIChatException', () {
      expect(const NetworkException('timeout'), isA<AIChatException>());
    });

    test('implements Exception', () {
      expect(const NetworkException('test'), isA<Exception>());
    });

    test('toString() includes cause when provided', () {
      final ex = NetworkException(
        'Connection refused',
        cause: 'SocketException',
      );
      expect(ex.toString(), contains('Connection refused'));
      expect(ex.toString(), contains('SocketException'));
    });

    test('toString() omits cause section when cause is null', () {
      final ex = NetworkException('Timeout');
      expect(ex.toString(), isNot(contains('Caused by')));
    });
  });

  group('RateLimitException', () {
    test('is a subtype of AIChatException', () {
      expect(
        const RateLimitException('Too many requests'),
        isA<AIChatException>(),
      );
    });

    test('implements Exception', () {
      expect(const RateLimitException('test'), isA<Exception>());
    });

    test('default statusCode is 429', () {
      const ex = RateLimitException('Limit exceeded');
      expect(ex.statusCode, 429);
    });

    test('stores retryAfter duration', () {
      final ex = RateLimitException(
        'Slow down',
        retryAfter: Duration(seconds: 30),
      );
      expect(ex.retryAfter, const Duration(seconds: 30));
    });

    test('retryAfter is null when not provided', () {
      const ex = RateLimitException('Limit hit');
      expect(ex.retryAfter, isNull);
    });

    test('toString() includes retry seconds when retryAfter is set', () {
      final ex = RateLimitException(
        'Rate limited',
        retryAfter: Duration(seconds: 60),
      );
      expect(ex.toString(), contains('60'));
    });

    test('toString() omits retry info when retryAfter is null', () {
      const ex = RateLimitException('Limit hit');
      expect(ex.toString(), isNot(contains('Retry after')));
    });
  });
}
