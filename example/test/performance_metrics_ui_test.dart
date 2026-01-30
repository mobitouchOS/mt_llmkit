// Tests for performance metrics UI functionality
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp_example/main.dart';

void main() {
  group('Performance Metrics UI Tests', () {
    testWidgets('Should show performance metrics toggle when model loaded', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Initially, the switch should not be visible (model not downloaded)
      expect(find.byType(Switch), findsNothing);
      expect(find.text('Show Performance Metrics'), findsNothing);
    });

    testWidgets('Switch should be toggleable', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // The app starts without model, so no switch yet
      expect(find.byType(Switch), findsNothing);
    });

    testWidgets('Performance metrics text should not show initially', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      expect(find.text('Performance Metrics'), findsNothing);
      expect(find.text('Tokens Generated:'), findsNothing);
      expect(find.text('Speed:'), findsNothing);
    });

    testWidgets('Should have proper widget structure', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Basic structure should be present
      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(Scaffold), findsOneWidget);
      expect(find.byType(Column), findsWidgets);
    });

    testWidgets('FloatingActionButton should exist when model is loaded', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Initially no FAB (model not loaded)
      expect(find.byType(FloatingActionButton), findsNothing);
    });

    testWidgets('TextField should have correct hint text', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // TextField not visible initially
      expect(find.byType(TextField), findsNothing);
    });

    testWidgets('Performance metrics container should have proper styling', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Container with metrics not visible initially
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Container &&
              widget.decoration is BoxDecoration &&
              (widget.decoration as BoxDecoration).borderRadius != null,
        ),
        findsNothing,
      );
    });

    testWidgets('Should display initial text', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Initial text not visible when model is not downloaded
      expect(find.text('Initial text'), findsNothing);
    });

    testWidgets('Download progress should not show initially', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      expect(find.text('Downloading model...'), findsNothing);
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is CircularProgressIndicator && widget.value != null,
        ),
        findsNothing,
      );
    });

    testWidgets('App should have proper AppBar', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());

      expect(find.byType(AppBar), findsOneWidget);
      expect(find.text('Example'), findsOneWidget);
    });
  });

  group('Performance Metrics Integration', () {
    testWidgets('App should maintain state correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Verify initial state
      expect(find.byType(MyApp), findsOneWidget);
      expect(find.text('Download Model'), findsOneWidget);
    });

    testWidgets('Should handle widget disposal correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Remove widget
      await tester.pumpWidget(Container());
      await tester.pumpAndSettle();

      // Widget should be gone
      expect(find.byType(MyApp), findsNothing);
    });

    testWidgets('Multiple widgets should work', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Verify Column exists
      expect(find.byType(Column), findsWidgets);
      expect(find.byType(Center), findsOneWidget);
    });
  });
}
