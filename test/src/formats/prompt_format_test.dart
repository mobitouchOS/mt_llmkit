import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';

void main() {
  group('PromptFormat Integration', () {
    test('ChatMLFormat should format prompt correctly', () {
      final format = ChatMLFormat();
      final result = format.formatPrompt('Hello, world!');

      expect(result, isNotEmpty);
      expect(result, contains('Hello, world!'));
    });

    test('AlpacaFormat should format prompt correctly', () {
      final format = AlpacaFormat();
      final result = format.formatPrompt('Test prompt');

      expect(result, isNotEmpty);
      expect(result, contains('Test prompt'));
    });

    test('GemmaFormat should format prompt correctly', () {
      final format = GemmaFormat();
      final result = format.formatPrompt('Gemma test');

      expect(result, isNotEmpty);
      expect(result, contains('Gemma test'));
    });
  });

  group('LlmConfig with different PromptFormats', () {
    test('should use ChatML format by default', () {
      const config = LlmConfig();

      expect(config.promptFormatDefault, isA<ChatMLFormat>());
    });

    test('should use custom AlpacaFormat', () {
      final format = AlpacaFormat();
      final config = LlmConfig(promptFormat: format);

      expect(config.promptFormatDefault, equals(format));
    });

    test('should use custom GemmaFormat', () {
      final format = GemmaFormat();
      final config = LlmConfig(promptFormat: format);

      expect(config.promptFormatDefault, equals(format));
    });
  });

  group('Model with different formats', () {
    test('LlmModelStandard should accept different formats', () {
      final chatmlModel = LlmModelStandard(
        LlmConfig(promptFormat: ChatMLFormat()),
      );
      final alpacaModel = LlmModelStandard(
        LlmConfig(promptFormat: AlpacaFormat()),
      );
      final gemmaModel = LlmModelStandard(
        LlmConfig(promptFormat: GemmaFormat()),
      );

      expect(chatmlModel, isNotNull);
      expect(alpacaModel, isNotNull);
      expect(gemmaModel, isNotNull);
    });

    test('LlmModelIsolated should accept different formats', () {
      final chatmlModel = LlmModelIsolated(
        LlmConfig(promptFormat: ChatMLFormat()),
      );
      final alpacaModel = LlmModelIsolated(
        LlmConfig(promptFormat: AlpacaFormat()),
      );
      final gemmaModel = LlmModelIsolated(
        LlmConfig(promptFormat: GemmaFormat()),
      );

      expect(chatmlModel, isNotNull);
      expect(alpacaModel, isNotNull);
      expect(gemmaModel, isNotNull);
    });
  });
}
