// Prompt format tests removed — PromptFormat system was dropped in the
// migration to llamadart, which applies chat templates automatically.
import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';

void main() {
  group('LlmConfig - prompt (llamadart chat templates)', () {
    test('models accept LlmConfig without promptFormat', () {
      final standard = LlmModelStandard(const LlmConfig());
      final isolated = LlmModelIsolated(const LlmConfig());

      expect(standard, isNotNull);
      expect(isolated, isNotNull);

      standard.dispose();
      isolated.dispose();
    });
  });
}
