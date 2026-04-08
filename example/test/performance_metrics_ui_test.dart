import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit_example/main.dart';

void main() {
  group('Performance Metrics UI Tests', () {
    testWidgets('No metrics text visible initially', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      // Metrics row only appears after a generation completes
      expect(find.textContaining('t/s'), findsNothing);
      expect(find.textContaining('ms/token'), findsNothing);
    });

    testWidgets('No CircularProgressIndicator before generation starts', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(CircularProgressIndicator), findsNothing);
    });

    testWidgets('No LinearProgressIndicator before download starts', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(LinearProgressIndicator), findsNothing);
    });

    testWidgets('Send button not visible when model is not ready', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byIcon(Icons.send), findsNothing);
    });

    testWidgets('Stop button not visible initially', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byIcon(Icons.stop_circle_outlined), findsNothing);
    });
  });

  group('Provider Selector Tests', () {
    testWidgets('SegmentedButton with Local GGUF and Rest API', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.text('Local GGUF'), findsOneWidget);
      expect(find.text('Rest API'), findsOneWidget);
    });

    testWidgets('Switching to Rest API hides download button', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.text('Download GGUF model (~800MB)'), findsOneWidget);

      await tester.tap(find.text('Rest API'));
      await tester.pumpAndSettle();
      expect(find.text('Download GGUF model (~800MB)'), findsNothing);
    });

    testWidgets('Switching back to Local GGUF restores download button', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('Rest API'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Local GGUF'));
      await tester.pumpAndSettle();
      expect(find.text('Download GGUF model (~800MB)'), findsOneWidget);
    });
  });
}
