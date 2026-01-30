// Tests for GenerationResult class
import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';

void main() {
  group('GenerationResult', () {
    test('should create successful result', () {
      const text = 'Generated text';
      final result = GenerationResult.success(text: text);

      expect(result.text, text);
      expect(result.isComplete, true);
      expect(result.isSuccess, true);
      expect(result.error, null);
    });

    test('should create successful result with metrics', () {
      const text = 'Generated text';
      final metrics = PerformanceMetrics(
        tokensGenerated: 10,
        durationMs: 1000,
        tokensPerSecond: 10.0,
        msPerToken: 100.0,
        startTime: DateTime.now(),
        endTime: DateTime.now(),
      );

      final result = GenerationResult.success(text: text, metrics: metrics);

      expect(result.text, text);
      expect(result.metrics, metrics);
      expect(result.isSuccess, true);
    });

    test('should create failed result', () {
      const errorMsg = 'Something went wrong';
      final result = GenerationResult.failure(error: errorMsg);

      expect(result.text, '');
      expect(result.error, errorMsg);
      expect(result.isComplete, false);
      expect(result.isSuccess, false);
    });

    test('should create failed result with partial text', () {
      const errorMsg = 'Connection lost';
      const partialText = 'Partial ';

      final result = GenerationResult.failure(
        error: errorMsg,
        text: partialText,
      );

      expect(result.text, partialText);
      expect(result.error, errorMsg);
      expect(result.isSuccess, false);
    });

    test('should convert successful result to string', () {
      const text = 'Generated text with some content';
      final result = GenerationResult.success(text: text);

      final str = result.toString();
      expect(str, contains('text length: ${text.length}'));
    });

    test('should convert failed result to string', () {
      const errorMsg = 'Test error';
      final result = GenerationResult.failure(error: errorMsg);

      final str = result.toString();
      expect(str, contains('error: $errorMsg'));
    });

    test('should convert to and from JSON without metrics', () {
      const text = 'Test text';
      final original = GenerationResult.success(text: text);

      final json = original.toJson();
      final restored = GenerationResult.fromJson(json);

      expect(restored.text, original.text);
      expect(restored.isComplete, original.isComplete);
      expect(restored.error, original.error);
      expect(restored.metrics, null);
    });

    test('should convert to and from JSON with metrics', () {
      const text = 'Test text';
      final metrics = PerformanceMetrics(
        tokensGenerated: 10,
        durationMs: 1000,
        tokensPerSecond: 10.0,
        msPerToken: 100.0,
        startTime: DateTime(2024, 1, 1, 12, 0, 0),
        endTime: DateTime(2024, 1, 1, 12, 0, 1),
      );

      final original = GenerationResult.success(text: text, metrics: metrics);

      final json = original.toJson();
      final restored = GenerationResult.fromJson(json);

      expect(restored.text, original.text);
      expect(restored.isComplete, original.isComplete);
      expect(restored.metrics!.tokensGenerated, metrics.tokensGenerated);
      expect(restored.metrics!.durationMs, metrics.durationMs);
      expect(restored.metrics!.tokensPerSecond, metrics.tokensPerSecond);
    });

    test('should convert failed result to and from JSON', () {
      const errorMsg = 'Test error';
      const text = 'Partial';

      final original = GenerationResult.failure(error: errorMsg, text: text);

      final json = original.toJson();
      final restored = GenerationResult.fromJson(json);

      expect(restored.text, text);
      expect(restored.error, errorMsg);
      expect(restored.isComplete, false);
      expect(restored.isSuccess, false);
    });

    test('isSuccess should be false when error exists', () {
      final result = GenerationResult(
        text: 'Some text',
        isComplete: true,
        error: 'Error occurred',
      );

      expect(result.isSuccess, false);
    });

    test('isSuccess should be false when not complete', () {
      final result = GenerationResult(text: 'Some text', isComplete: false);

      expect(result.isSuccess, false);
    });

    test('isSuccess should be true when complete and no error', () {
      final result = GenerationResult(text: 'Some text', isComplete: true);

      expect(result.isSuccess, true);
    });
  });
}
