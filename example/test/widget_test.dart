import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit_example/main.dart';

void main() {
  group('MyApp Widget Tests', () {
    testWidgets('Should build MaterialApp with MainPage', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.byType(MaterialApp), findsOneWidget);
      expect(find.byType(MainPage), findsOneWidget);
    });

    testWidgets('AppBar shows LLM title on initial tab', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.byType(AppBar), findsOneWidget);
      expect(find.text('llmcpp — LLM Demo'), findsOneWidget);
    });

    testWidgets('NavigationBar has three tabs', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.byType(NavigationBar), findsOneWidget);
      expect(find.text('LLM'), findsOneWidget);
      expect(find.text('Vision'), findsOneWidget);
      expect(find.text('RAG'), findsOneWidget);
    });

    testWidgets('LlmDemoPage shows provider SegmentedButton', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.text('Local GGUF'), findsOneWidget);
      expect(find.text('Rest API'), findsOneWidget);
    });

    testWidgets('Download button visible when model not ready', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.text('Download GGUF model (~800MB)'), findsOneWidget);
    });

    testWidgets('No TextField visible when model not ready', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(TextField), findsNothing);
    });
  });

  group('Navigation Tests', () {
    testWidgets('Tapping Vision tab changes AppBar title', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('Vision'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — Vision Demo'), findsOneWidget);
    });

    testWidgets('Tapping RAG tab changes AppBar title', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('RAG'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — RAG Demo'), findsOneWidget);
    });

    testWidgets('Tapping back to LLM tab restores title', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('Vision'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('LLM'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — LLM Demo'), findsOneWidget);
    });
  });
}
