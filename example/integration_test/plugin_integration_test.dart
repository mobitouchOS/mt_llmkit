// This is a basic Flutter integration test.
//
// Since integration tests run in a full Flutter application, they can interact
// with the host side of a plugin implementation, unlike Dart unit tests.
//
// For more information about Flutter integration tests, please see
// https://flutter.dev/to/integration-testing

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/models/llm_model_isolated.dart';
import 'package:mt_llmkit/src/models/llm_model_standard.dart';
import 'package:mt_llmkit_example/main.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Plugin Integration Tests', () {
    test('LlmModelIsolated should be instantiable', () {
      final model = LlmModelIsolated(const LlmConfig());
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
      model.dispose();
      expect(model.isDisposed, true);
    });

    test('LlmModelStandard should be instantiable', () {
      final model = LlmModelStandard(const LlmConfig());
      expect(model, isNotNull);
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
      model.dispose();
      expect(model.isDisposed, true);
    });

    test('LlmConfig should have default values', () {
      const config = LlmConfig();
      expect(config, isNotNull);
      expect(config.nGpuLayersDefault, isA<int>());
      expect(config.nCtxDefault, isA<int>());
      expect(config.tempDefault, isA<double>());
    });

    test('LlmConfig should accept custom values', () {
      const config = LlmConfig(
        nGpuLayers: 4,
        nCtx: 2048,
        nThreads: 4,
        nPredict: 256,
        temp: 0.7,
        topP: 0.9,
        penaltyRepeat: 1.1,
      );
      expect(config.nGpuLayersDefault, 4);
      expect(config.nCtxDefault, 2048);
      expect(config.nThreadsDefault, 4);
      expect(config.nPredictDefault, 256);
      expect(config.tempDefault, 0.7);
      expect(config.topPDefault, 0.9);
      expect(config.penaltyRepeatDefault, 1.1);
    });

    test('Should support different configs', () {
      final model1 = LlmModelIsolated(const LlmConfig(temp: 0.5));
      final model2 = LlmModelIsolated(const LlmConfig(nGpuLayers: 0));
      final model3 = LlmModelIsolated(const LlmConfig(nCtx: 4096));

      expect(model1, isNotNull);
      expect(model2, isNotNull);
      expect(model3, isNotNull);

      model1.dispose();
      model2.dispose();
      model3.dispose();
    });

    test('Model should throw error on operations after dispose', () {
      final model = LlmModelIsolated(const LlmConfig());
      model.dispose();

      expect(() => model.loadModel('/test.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });

    test('Model should throw error on prompt before initialization', () {
      final model = LlmModelIsolated(const LlmConfig());
      expect(() => model.sendPrompt('test'), throwsStateError);
      model.dispose();
    });

    test('Multiple models can coexist', () {
      final model1 = LlmModelIsolated(const LlmConfig());
      final model2 = LlmModelStandard(const LlmConfig());

      expect(model1.isInitialized, false);
      expect(model2.isInitialized, false);

      model1.dispose();
      model2.dispose();

      expect(model1.isDisposed, true);
      expect(model2.isDisposed, true);
    });
  });

  group('UI Integration Tests', () {
    testWidgets('MyApp should render without errors', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(Scaffold), findsOneWidget);
      expect(find.text('Example'), findsOneWidget);
    });

    testWidgets('Download button should be visible initially', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      expect(find.text('Download Model'), findsOneWidget);
      expect(find.byType(ElevatedButton), findsOneWidget);
    });

    testWidgets('UI should be responsive to user interactions', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      final downloadButton = find.byType(ElevatedButton);
      expect(downloadButton, findsOneWidget);

      // Verify button has onPressed handler
      final ElevatedButton button = tester.widget(downloadButton);
      expect(button.onPressed, isNotNull);
    });

    testWidgets('AppBar should display correct title', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      expect(find.text('Example'), findsOneWidget);
      expect(
        find.descendant(
          of: find.byType(AppBar),
          matching: find.text('Example'),
        ),
        findsOneWidget,
      );
    });

    testWidgets('Center widget should contain main content', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      expect(find.byType(Center), findsOneWidget);
      expect(
        find.descendant(of: find.byType(Center), matching: find.byType(Column)),
        findsOneWidget,
      );
    });
  });

  group('State Management Integration Tests', () {
    testWidgets('App should maintain state correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      // Initial state - model not downloaded
      expect(find.text('Download Model'), findsOneWidget);
      expect(find.byType(TextField), findsNothing);
      expect(find.byType(FloatingActionButton), findsNothing);
    });

    testWidgets('App should dispose resources properly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpAndSettle();

      // Verify app builds successfully
      expect(find.byType(MyApp), findsOneWidget);

      // Remove widget tree to trigger dispose
      await tester.pumpWidget(Container());
      await tester.pumpAndSettle();

      // App should dispose without errors
      expect(find.byType(MyApp), findsNothing);
    });
  });
}
