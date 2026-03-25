import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';

void main() {
  group('llmcpp package exports', () {
    test('should export LlmConfig', () {
      const config = LlmConfig();
      expect(config, isNotNull);
    });

    test('should export LlmInterface', () {
      expect(LlmInterface, isNotNull);
    });

    test('should export LocalModel', () {
      final model = LocalModel();
      expect(model, isNotNull);
      model.dispose();
    });

    test('should export ModelBackend', () {
      expect(ModelBackend.isolate, isNotNull);
      expect(ModelBackend.inProcess, isNotNull);
    });

    test('should export LlamaImageContent', () {
      const image = LlamaImageContent(path: '/test/image.jpg');
      expect(image, isNotNull);
    });

    test('should export RagEngine', () {
      expect(RagEngine, isNotNull);
    });
  });
}
