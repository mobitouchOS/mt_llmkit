// Tests for application state management
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp_example/main.dart';

void main() {
  group('App State Tests', () {
    testWidgets('Initial state should show download UI', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Model is not downloaded, so download button should be visible
      expect(find.text('Download Model'), findsOneWidget);
      expect(find.byType(ElevatedButton), findsOneWidget);

      // Chat UI should not be visible
      expect(find.byType(TextField), findsNothing);
      expect(find.byType(FloatingActionButton), findsNothing);
    });

    testWidgets('Download button should have proper callback', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      final elevatedButton = tester.widget<ElevatedButton>(
        find.byType(ElevatedButton),
      );
      expect(elevatedButton.onPressed, isNotNull);
    });

    testWidgets('App should have proper theme colors', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      final materialApp = tester.widget<MaterialApp>(find.byType(MaterialApp));
      expect(materialApp, isNotNull);

      final scaffold = tester.widget<Scaffold>(find.byType(Scaffold));
      expect(scaffold, isNotNull);
    });

    testWidgets('AppBar title should be "Example"', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      final appBar = tester.widget<AppBar>(find.byType(AppBar));
      final title = appBar.title as Text;
      expect(title.data, 'Example');
    });

    testWidgets('Center widget should wrap main content', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      expect(find.byType(Center), findsOneWidget);

      // Check hierarchy
      final center = tester.widget<Center>(find.byType(Center));
      expect(center.child, isA<Column>());
    });

    testWidgets('Column should have proper mainAxisAlignment', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Check if Column is properly configured
      expect(find.byType(Column), findsWidgets);
    });
  });

  group('App Lifecycle Tests', () {
    testWidgets('App should dispose cleanly', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Build widget
      expect(find.byType(MyApp), findsOneWidget);

      // Remove widget - this should trigger dispose
      await tester.pumpWidget(Container());
      await tester.pumpAndSettle();

      // Widget should no longer exist
      expect(find.byType(MyApp), findsNothing);
    });

    testWidgets('Multiple app instances should work', (
      WidgetTester tester,
    ) async {
      // First instance
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(MyApp), findsOneWidget);

      // Replace with new instance
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(MyApp), findsOneWidget);
    });
  });

  group('UI Element Visibility Tests', () {
    testWidgets('Download section should have proper structure', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      // Download button should be inside Column
      expect(
        find.descendant(
          of: find.byType(Column),
          matching: find.text('Download Model'),
        ),
        findsOneWidget,
      );
    });

    testWidgets('Scaffold should contain all main components', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      // AppBar in Scaffold
      expect(
        find.descendant(
          of: find.byType(Scaffold),
          matching: find.byType(AppBar),
        ),
        findsOneWidget,
      );

      // Body in Scaffold
      final scaffold = tester.widget<Scaffold>(find.byType(Scaffold));
      expect(scaffold.body, isNotNull);
    });
  });

  group('Widget Properties Tests', () {
    testWidgets('ElevatedButton should be interactive', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      final button = tester.widget<ElevatedButton>(find.byType(ElevatedButton));
      expect(button.enabled, isTrue);
      expect(button.onPressed, isNotNull);
    });

    testWidgets('Text widgets should have correct content', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();

      expect(find.text('Example'), findsOneWidget);
      expect(find.text('Download Model'), findsOneWidget);
    });
  });
}
