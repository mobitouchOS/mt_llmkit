// test/src/api/ai_chat_provider_factory_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/src/api/ai_chat_provider.dart';
import 'package:mt_llmkit/src/api/ai_chat_provider_factory.dart';
import 'package:mt_llmkit/src/api/claude_chat_provider.dart';
import 'package:mt_llmkit/src/api/gemini_chat_provider.dart';
import 'package:mt_llmkit/src/api/mistral_chat_provider.dart';
import 'package:mt_llmkit/src/api/openai_chat_provider.dart';

void main() {
  // ── Factory.create ─────────────────────────────────────────────────────────

  group('AIChatProviderFactory.create', () {
    test('returns OpenAIChatProvider for openai', () {
      expect(
        AIChatProviderFactory.create(AIChatProviderType.openai),
        isA<OpenAIChatProvider>(),
      );
    });

    test('returns GeminiChatProvider for gemini', () {
      expect(
        AIChatProviderFactory.create(AIChatProviderType.gemini),
        isA<GeminiChatProvider>(),
      );
    });

    test('returns ClaudeChatProvider for claude', () {
      expect(
        AIChatProviderFactory.create(AIChatProviderType.claude),
        isA<ClaudeChatProvider>(),
      );
    });

    test('returns MistralChatProvider for mistral', () {
      expect(
        AIChatProviderFactory.create(AIChatProviderType.mistral),
        isA<MistralChatProvider>(),
      );
    });

    test('every provider implements AIChatProvider', () {
      for (final type in AIChatProviderType.values) {
        expect(
          AIChatProviderFactory.create(type),
          isA<AIChatProvider>(),
        );
      }
    });

    test('newly created providers are not initialised', () {
      for (final type in AIChatProviderType.values) {
        expect(AIChatProviderFactory.create(type).isInitialized, isFalse);
      }
    });
  });

  // ── Lifecycle: not initialised ─────────────────────────────────────────────

  group('Provider lifecycle — not initialised', () {
    test('sendMessage throws StateError before initialize()', () {
      final provider = OpenAIChatProvider();
      expect(() => provider.sendMessage('Hello'), throwsStateError);
    });

    test('sendChatMessages throws StateError before initialize()', () {
      final provider = ClaudeChatProvider();
      expect(() => provider.sendChatMessages([]), throwsStateError);
    });

    test('sendMessageStream throws StateError before initialize()', () {
      final provider = MistralChatProvider();
      expect(() => provider.sendMessageStream('Hi'), throwsStateError);
    });

    test('sendMessage throws StateError after dispose()', () async {
      final provider = OpenAIChatProvider();
      await provider.dispose();
      expect(() => provider.sendMessage('Hello'), throwsStateError);
    });
  });

  // ── Lifecycle: initialize() ────────────────────────────────────────────────

  group('Provider lifecycle — initialize()', () {
    test('sets isInitialized to true for all providers', () async {
      for (final type in AIChatProviderType.values) {
        final provider = AIChatProviderFactory.create(type);
        await provider.initialize({'apiKey': 'test-key'});
        expect(provider.isInitialized, isTrue,
            reason: '$type should be initialized');
        await provider.dispose();
      }
    });

    test('throws ArgumentError when apiKey is absent', () async {
      for (final type in AIChatProviderType.values) {
        await expectLater(
          AIChatProviderFactory.create(type).initialize({}),
          throwsArgumentError,
          reason: '$type should reject empty config',
        );
      }
    });

    test('throws ArgumentError when apiKey is an empty string', () async {
      for (final type in AIChatProviderType.values) {
        await expectLater(
          AIChatProviderFactory.create(type).initialize({'apiKey': ''}),
          throwsArgumentError,
          reason: '$type should reject empty apiKey',
        );
      }
    });

    test('dispose() resets isInitialized to false', () async {
      final provider = OpenAIChatProvider();
      await provider.initialize({'apiKey': 'sk-test'});
      expect(provider.isInitialized, isTrue);
      await provider.dispose();
      expect(provider.isInitialized, isFalse);
    });

    test('calling dispose() twice does not throw', () async {
      final provider = GeminiChatProvider();
      await provider.initialize({'apiKey': 'test-key'});
      await provider.dispose();
      await expectLater(provider.dispose(), completes);
    });
  });

  // ── AIChatProviderType enum ────────────────────────────────────────────────

  group('AIChatProviderType', () {
    test('has exactly four values', () {
      expect(AIChatProviderType.values.length, 4);
    });

    test('contains all expected provider types', () {
      expect(AIChatProviderType.values, contains(AIChatProviderType.openai));
      expect(AIChatProviderType.values, contains(AIChatProviderType.gemini));
      expect(AIChatProviderType.values, contains(AIChatProviderType.claude));
      expect(AIChatProviderType.values, contains(AIChatProviderType.mistral));
    });
  });
}
