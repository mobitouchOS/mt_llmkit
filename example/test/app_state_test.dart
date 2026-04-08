import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit_example/main.dart';

void main() {
  group('App State Tests', () {
    testWidgets('Initial tab is LLM (index 0)', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.text('llmcpp — LLM Demo'), findsOneWidget);
      expect(find.text('Local GGUF'), findsOneWidget);
    });

    testWidgets('Switching to Vision tab updates state', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('Vision'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — Vision Demo'), findsOneWidget);
      expect(find.text('llmcpp — LLM Demo'), findsNothing);
    });

    testWidgets('Switching to RAG tab updates state', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('RAG'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — RAG Demo'), findsOneWidget);
      expect(find.text('llmcpp — LLM Demo'), findsNothing);
    });

    testWidgets('Tab navigation is persistent across switches', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.tap(find.text('Vision'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('RAG'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('LLM'));
      await tester.pumpAndSettle();
      expect(find.text('llmcpp — LLM Demo'), findsOneWidget);
    });
  });

  group('App Lifecycle Tests', () {
    testWidgets('App disposes cleanly', (tester) async {
      await tester.pumpWidget(const MyApp());
      expect(find.byType(MyApp), findsOneWidget);
      await tester.pumpWidget(Container());
      await tester.pumpAndSettle();
      expect(find.byType(MyApp), findsNothing);
    });

    testWidgets('App rebuilds correctly after replace', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(MyApp), findsOneWidget);
    });
  });

  group('LLM Page State Tests', () {
    testWidgets('Download button visible in initial GGUF state', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.text('Download GGUF model (~800MB)'), findsOneWidget);
    });

    testWidgets('No TextField visible before model is ready', (tester) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(TextField), findsNothing);
    });

    testWidgets('No LinearProgressIndicator visible before download starts', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());
      await tester.pump();
      expect(find.byType(LinearProgressIndicator), findsNothing);
    });
  });
}
