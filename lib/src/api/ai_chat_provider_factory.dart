// lib/src/api/ai_chat_provider_factory.dart

import 'ai_chat_provider.dart';
import 'claude_chat_provider.dart';
import 'gemini_chat_provider.dart';
import 'mistral_chat_provider.dart';
import 'openai_chat_provider.dart';

/// Supported AI chat provider backends.
enum AIChatProviderType {
  /// OpenAI Chat Completions API (`gpt-4o-mini` by default).
  openai,

  /// Google Gemini API (`gemini-1.5-flash` by default).
  gemini,

  /// Anthropic Claude Messages API (`claude-haiku-4-5-20251001` by default).
  claude,

  /// Mistral AI Chat Completions API (`mistral-small-latest` by default).
  mistral,
}

/// Factory for creating [AIChatProvider] instances.
///
/// Callers program against the [AIChatProvider] abstraction, satisfying the
/// Dependency Inversion Principle — no concrete provider class needs to be
/// imported by business logic.
///
/// ## Create only (manual lifecycle):
/// ```dart
/// final provider = AIChatProviderFactory.create(AIChatProviderType.openai);
/// await provider.initialize({'apiKey': 'sk-...'});
///
/// final response = await provider.sendMessage('Hello!');
/// print(response.message.content);
///
/// await provider.dispose();
/// ```
///
/// ## Create + initialize in one step:
/// ```dart
/// final provider = await AIChatProviderFactory.createAndInitialize(
///   AIChatProviderType.gemini,
///   {'apiKey': 'AIza...'},
/// );
///
/// await for (final token in provider.sendMessageStream('Explain FP.')) {
///   stdout.write(token);
/// }
///
/// await provider.dispose();
/// ```
///
/// ## Swap providers without changing business logic:
/// ```dart
/// Future<String> ask(AIChatProvider provider, String question) async {
///   final r = await provider.sendMessage(question);
///   return r.message.content;
/// }
///
/// // Switch from Claude to Mistral with no change to ask():
/// final provider = await AIChatProviderFactory.createAndInitialize(
///   AIChatProviderType.mistral,
///   {'apiKey': '...'},
/// );
/// print(await ask(provider, 'What is a monad?'));
/// ```
class AIChatProviderFactory {
  // Prevent instantiation — this is a pure static factory.
  AIChatProviderFactory._();

  /// Creates an uninitialised [AIChatProvider] of the given [type].
  ///
  /// Call [AIChatProvider.initialize] with provider-specific config before
  /// using the returned instance.
  static AIChatProvider create(AIChatProviderType type) {
    return switch (type) {
      AIChatProviderType.openai => OpenAIChatProvider(),
      AIChatProviderType.gemini => GeminiChatProvider(),
      AIChatProviderType.claude => ClaudeChatProvider(),
      AIChatProviderType.mistral => MistralChatProvider(),
    };
  }

  /// Creates and initialises an [AIChatProvider] in a single step.
  ///
  /// [config] is forwarded directly to [AIChatProvider.initialize].
  /// Throws [ArgumentError] if required config keys are missing.
  static Future<AIChatProvider> createAndInitialize(
    AIChatProviderType type,
    Map<String, dynamic> config,
  ) async {
    final provider = create(type);
    await provider.initialize(config);
    return provider;
  }
}
