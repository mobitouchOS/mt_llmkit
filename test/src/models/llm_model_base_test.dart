import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';

/// Mock implementation for testing purposes
class MockLlmModel extends LlmModelBase {
  bool _loadModelCalled = false;
  bool _sendPromptCalled = false;
  bool _disposeCalled = false;
  bool _cleanCalled = false;

  String? _lastPrompt;
  String? _lastModelPath;

  bool get loadModelCalled => _loadModelCalled;
  bool get sendPromptCalled => _sendPromptCalled;
  bool get disposeCalled => _disposeCalled;
  bool get cleanCalled => _cleanCalled;

  String? get lastPrompt => _lastPrompt;
  String? get lastModelPath => _lastModelPath;

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();
    _loadModelCalled = true;
    _lastModelPath = localPath;
    markAsInitialized();
  }

  @override
  Stream<String>? sendPrompt(String prompt) {
    checkInitialized();
    _sendPromptCalled = true;
    _lastPrompt = prompt;
    return Stream.value('Mock response for: $prompt');
  }

  @override
  Future<GenerationResult> sendPromptWithMetrics(String prompt) async {
    checkInitialized();
    _sendPromptCalled = true;
    _lastPrompt = prompt;

    final startTime = DateTime.now();
    final text = 'Mock response for: $prompt';
    final endTime = DateTime.now();

    final metrics = PerformanceMetrics.fromGeneration(
      tokenCount: 10,
      startTime: startTime,
      endTime: endTime,
    );

    return GenerationResult.success(text: text, metrics: metrics);
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) async* {
    checkInitialized();
    _sendPromptCalled = true;
    _lastPrompt = prompt;

    final startTime = DateTime.now();
    final text = 'Mock response for: $prompt';

    // Simulate streaming with metrics
    for (var i = 0; i < text.length; i += 5) {
      final chunk = text.substring(i, (i + 5).clamp(0, text.length));
      final currentTime = DateTime.now();
      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: i ~/ 5 + 1,
        startTime: startTime,
        endTime: currentTime,
      );

      yield StreamingChunk(
        text: chunk,
        metrics: metrics,
        isFinal: i + 5 >= text.length,
      );
    }
  }

  @override
  void dispose() {
    _disposeCalled = true;
    markAsDisposed();
  }

  @override
  void clean() {
    checkInitialized();
    _cleanCalled = true;
  }
}

void main() {
  group('LlmModelBase', () {
    late MockLlmModel model;

    setUp(() {
      model = MockLlmModel();
    });

    test('should initialize with correct default state', () {
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
    });

    test('should mark as initialized after loading', () async {
      await model.loadModel('/test/path');

      expect(model.isInitialized, true);
      expect(model.isDisposed, false);
      expect(model.loadModelCalled, true);
      expect(model.lastModelPath, '/test/path');
    });

    test('should mark as disposed after dispose', () async {
      await model.loadModel('/test/path');
      model.dispose();

      expect(model.isInitialized, false);
      expect(model.isDisposed, true);
      expect(model.disposeCalled, true);
    });

    test(
      'should throw StateError when sending prompt before initialization',
      () {
        expect(() => model.sendPrompt('test'), throwsStateError);
      },
    );

    test('should throw StateError when cleaning before initialization', () {
      expect(() => model.clean(), throwsStateError);
    });

    test('should allow sending prompt after initialization', () async {
      await model.loadModel('/test/path');
      final stream = model.sendPrompt('test prompt');

      expect(stream, isNotNull);
      expect(model.sendPromptCalled, true);
      expect(model.lastPrompt, 'test prompt');

      final response = await stream!.first;
      expect(response, 'Mock response for: test prompt');
    });

    test('should allow cleaning after initialization', () async {
      await model.loadModel('/test/path');
      model.clean();

      expect(model.cleanCalled, true);
    });

    test('should throw StateError when loading after dispose', () async {
      await model.loadModel('/test/path');
      model.dispose();

      expect(() => model.loadModel('/test/path2'), throwsStateError);
    });

    test('should throw StateError when sending prompt after dispose', () async {
      await model.loadModel('/test/path');
      model.dispose();

      expect(() => model.sendPrompt('test'), throwsStateError);
    });

    test('should throw StateError when cleaning after dispose', () async {
      await model.loadModel('/test/path');
      model.dispose();

      expect(() => model.clean(), throwsStateError);
    });

    test('should handle multiple operations in sequence', () async {
      // Load
      await model.loadModel('/test/path1');
      expect(model.isInitialized, true);

      // Send prompt
      final stream1 = model.sendPrompt('prompt1');
      expect(stream1, isNotNull);
      await stream1!.first;

      // Clean
      model.clean();
      expect(model.cleanCalled, true);

      // Send another prompt
      final stream2 = model.sendPrompt('prompt2');
      expect(stream2, isNotNull);
      await stream2!.first;

      // Dispose
      model.dispose();
      expect(model.isDisposed, true);
    });
  });
}
