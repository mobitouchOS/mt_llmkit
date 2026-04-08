// Tests for LLM model functionality in the example app
import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/llmcpp.dart';
import 'package:mt_llmkit/src/models/llm_model_isolated.dart';
import 'package:mt_llmkit/src/models/llm_model_standard.dart';

void main() {
  group('LlmModelIsolated Basic Tests', () {
    test('Should create model with default config', () {
      final model = LlmModelIsolated(const LlmConfig());
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
      model.dispose();
    });

    test('Should create model with custom config', () {
      const config = LlmConfig(
        temp: 0.7,
        nGpuLayers: 4,
        nCtx: 2048,
        nThreads: 4,
        nPredict: 256,
        topP: 0.9,
        penaltyRepeat: 1.1,
      );
      final model = LlmModelIsolated(config);
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      model.dispose();
    });

    test('Should dispose model correctly', () {
      final model = LlmModelIsolated(const LlmConfig());
      expect(model.isDisposed, false);

      model.dispose();

      expect(model.isDisposed, true);
    });

    test('Should not allow operations after dispose', () {
      final model = LlmModelIsolated(const LlmConfig());
      model.dispose();

      expect(() => model.loadModel('/test/model.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });

    test('Should not allow prompt before initialization', () {
      final model = LlmModelIsolated(const LlmConfig());

      expect(() => model.sendPrompt('test'), throwsStateError);

      model.dispose();
    });

    test('Should not allow clean before initialization', () {
      final model = LlmModelIsolated(const LlmConfig());

      expect(() => model.clean(), throwsStateError);

      model.dispose();
    });
  });

  group('LlmModelStandard Basic Tests', () {
    test('Should create model with default config', () {
      final model = LlmModelStandard(const LlmConfig());
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
      model.dispose();
    });

    test('Should create model with custom config', () {
      const config = LlmConfig(
        temp: 0.7,
        nGpuLayers: 4,
        nCtx: 2048,
        nThreads: 4,
        nPredict: 256,
        topP: 0.9,
        penaltyRepeat: 1.1,
      );
      final model = LlmModelStandard(config);
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      model.dispose();
    });

    test('Should dispose model correctly', () {
      final model = LlmModelStandard(const LlmConfig());
      expect(model.isDisposed, false);

      model.dispose();

      expect(model.isDisposed, true);
    });

    test('Should not allow operations after dispose', () {
      final model = LlmModelStandard(const LlmConfig());
      model.dispose();

      expect(() => model.loadModel('/test/model.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });
  });

  group('LlmConfig Tests', () {
    test('Should have default values', () {
      const config = LlmConfig();
      expect(config.nGpuLayersDefault, isA<int>());
      expect(config.nCtxDefault, isA<int>());
      expect(config.nThreadsDefault, isA<int>());
      expect(config.nPredictDefault, isA<int>());
      expect(config.tempDefault, isA<double>());
      expect(config.topPDefault, isA<double>());
      expect(config.penaltyRepeatDefault, isA<double>());
    });

    test('Should accept custom values', () {
      const config = LlmConfig(
        nGpuLayers: 8,
        nCtx: 4096,
        nThreads: 8,
        nPredict: 512,
        temp: 0.8,
        topP: 0.95,
        penaltyRepeat: 1.2,
      );
      expect(config.nGpuLayersDefault, 8);
      expect(config.nCtxDefault, 4096);
      expect(config.nThreadsDefault, 8);
      expect(config.nPredictDefault, 512);
      expect(config.tempDefault, 0.8);
      expect(config.topPDefault, 0.95);
      expect(config.penaltyRepeatDefault, 1.2);
    });

    test('Should work with different configs', () {
      const chatmlConfig = LlmConfig(temp: 0.5);
      const alpacaConfig = LlmConfig(nGpuLayers: 0);
      const gemmaConfig = LlmConfig(nCtx: 4096);

      expect(chatmlConfig.tempDefault, 0.5);
      expect(alpacaConfig.nGpuLayersDefault, 0);
      expect(gemmaConfig.nCtxDefault, 4096);
    });
  });

  group('Multiple Models Tests', () {
    test('Should create multiple isolated models', () {
      final model1 = LlmModelIsolated(const LlmConfig());
      final model2 = LlmModelIsolated(const LlmConfig());

      expect(model1, isNotNull);
      expect(model2, isNotNull);
      expect(model1.isInitialized, false);
      expect(model2.isInitialized, false);

      model1.dispose();
      model2.dispose();

      expect(model1.isDisposed, true);
      expect(model2.isDisposed, true);
    });

    test('Should create multiple standard models', () {
      final model1 = LlmModelStandard(const LlmConfig());
      final model2 = LlmModelStandard(const LlmConfig());

      expect(model1, isNotNull);
      expect(model2, isNotNull);
      expect(model1.isInitialized, false);
      expect(model2.isInitialized, false);

      model1.dispose();
      model2.dispose();

      expect(model1.isDisposed, true);
      expect(model2.isDisposed, true);
    });

    test('Should create mixed model types', () {
      final isolated = LlmModelIsolated(const LlmConfig());
      final standard = LlmModelStandard(const LlmConfig());

      expect(isolated, isNotNull);
      expect(standard, isNotNull);
      expect(isolated.isInitialized, false);
      expect(standard.isInitialized, false);

      isolated.dispose();
      standard.dispose();

      expect(isolated.isDisposed, true);
      expect(standard.isDisposed, true);
    });
  });

  group('Model Lifecycle Tests', () {
    test('Should handle multiple dispose calls safely', () {
      final model = LlmModelIsolated(const LlmConfig());

      model.dispose();
      expect(model.isDisposed, true);

      // Second dispose should not cause an error
      expect(() => model.dispose(), returnsNormally);
    });

    test('Should maintain state correctly', () {
      final model = LlmModelIsolated(const LlmConfig());

      // Initial state
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);

      // After dispose
      model.dispose();
      expect(model.isDisposed, true);
    });
  });

  group('Error Handling Tests', () {
    test('Should throw StateError on operations after dispose', () {
      final model = LlmModelIsolated(const LlmConfig());
      model.dispose();

      expect(() => model.loadModel('/path/to/model.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test prompt'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });

    test('Should throw StateError on uninitialized operations', () {
      final model = LlmModelIsolated(const LlmConfig());

      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);

      model.dispose();
    });
  });
}
