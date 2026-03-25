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

    test('should export LlmModelBase', () {
      expect(LlmModelBase, isNotNull);
    });

    test('should export LlmModelStandard', () {
      final model = LlmModelStandard(const LlmConfig());
      expect(model, isNotNull);
      model.dispose();
    });

    test('should export LlmModelIsolated', () {
      final model = LlmModelIsolated(const LlmConfig());
      expect(model, isNotNull);
      model.dispose();
    });

    test('should export LlamaImageContent', () {
      const image = LlamaImageContent(path: '/test/image.jpg');
      expect(image, isNotNull);
    });
  });
}
