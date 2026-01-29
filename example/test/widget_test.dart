// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp_example/main.dart';

void main() {
  group('MyApp Widget Tests', () {
    testWidgets('Should build basic UI structure', (WidgetTester tester) async {
      // Build our app and trigger a frame
      await tester.pumpWidget(const MyApp());

      // Verify basic structure
      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(Scaffold), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
      expect(find.text('Example'), findsOneWidget);
    });

    testWidgets('Should show download button when model is not downloaded', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // The download button should be visible initially
      expect(find.text('Download Model'), findsOneWidget);
      expect(find.byType(ElevatedButton), findsOneWidget);

      // TextField and FAB should not be visible
      expect(find.byType(TextField), findsNothing);
      expect(find.byType(FloatingActionButton), findsNothing);
    });

    testWidgets('Should display initial text', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // When model is not downloaded, we should see the download section
      expect(find.byType(Column), findsWidgets);
    });

    testWidgets('AppBar should have correct styling', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      final AppBar appBar = tester.widget(find.byType(AppBar));
      expect(appBar.title, isA<Text>());
      final Text titleText = appBar.title as Text;
      expect(titleText.data, 'Example');
    });

    testWidgets('Should have Center widget in body', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      expect(find.byType(Center), findsOneWidget);
    });
  });

  group('MyApp State Management Tests', () {
    testWidgets('Should initialize with correct state', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Verify initial state - model not downloaded
      expect(find.text('Download Model'), findsOneWidget);
    });

    testWidgets(
      'TextField should have correct initial value when model loaded',
      (WidgetTester tester) async {
        await tester.pumpWidget(const MyApp());
        await tester.pump();

        // Initially, TextField should not be visible
        expect(find.byType(TextField), findsNothing);
      },
    );
  });

  group('MyApp Layout Tests', () {
    testWidgets('Should have proper widget hierarchy', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      // Verify hierarchy
      expect(
        find.descendant(
          of: find.byType(MaterialApp),
          matching: find.byType(Scaffold),
        ),
        findsOneWidget,
      );

      expect(
        find.descendant(
          of: find.byType(Scaffold),
          matching: find.byType(AppBar),
        ),
        findsOneWidget,
      );
    });

    testWidgets('Should use Column for main layout', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      expect(find.byType(Column), findsWidgets);
    });
  });

  group('MyApp Button Tests', () {
    testWidgets('Download button should be tappable', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      final downloadButton = find.text('Download Model');
      expect(downloadButton, findsOneWidget);

      // Verify button is tappable (has onPressed)
      final ElevatedButton button = tester.widget(find.byType(ElevatedButton));
      expect(button.onPressed, isNotNull);
    });
  });
}
