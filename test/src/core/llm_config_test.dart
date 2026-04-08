import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

void main() {
  group('LlmConfig', () {
    test('should create config with default values', () {
      const config = LlmConfig();

      expect(config.nGpuLayersDefault, 64);
      expect(config.nCtxDefault, 8192);
      expect(config.nBatchDefault, 4096);
      expect(config.nPredictDefault, 8192);
      expect(config.nThreadsDefault, 6);
      expect(config.tempDefault, 0.72);
      expect(config.topKDefault, 64);
      expect(config.topPDefault, 0.95);
      expect(config.penaltyRepeatDefault, 1.1);
    });

    test('should create config with custom values', () {
      const config = LlmConfig(
        nGpuLayers: 32,
        nCtx: 4096,
        nBatch: 2048,
        nPredict: 4096,
        nThreads: 4,
        temp: 0.8,
        topK: 40,
        topP: 0.9,
        penaltyRepeat: 1.2,
      );

      expect(config.nGpuLayersDefault, 32);
      expect(config.nCtxDefault, 4096);
      expect(config.nBatchDefault, 2048);
      expect(config.nPredictDefault, 4096);
      expect(config.nThreadsDefault, 4);
      expect(config.tempDefault, 0.8);
      expect(config.topKDefault, 40);
      expect(config.topPDefault, 0.9);
      expect(config.penaltyRepeatDefault, 1.2);
    });

    test('should handle null values gracefully', () {
      const config = LlmConfig(nGpuLayers: null, nCtx: null);

      expect(config.nGpuLayersDefault, 64);
      expect(config.nCtxDefault, 8192);
    });

    test('should preserve original null values', () {
      const config = LlmConfig();

      expect(config.nGpuLayers, null);
      expect(config.nCtx, null);
      expect(config.nBatch, null);
    });
  });
}
