// lib/src/api/chat_models.dart

/// Role of a participant in a chat conversation.
enum ChatRole {
  /// End-user turn.
  user,

  /// Model (AI) turn.
  assistant,

  /// Invisible instruction that shapes model behaviour.
  system,
}

/// A single message in a chat conversation.
class ChatMessage {
  const ChatMessage({required this.role, required this.content});

  /// Shorthand for a user message.
  const ChatMessage.user(String content)
      : this(role: ChatRole.user, content: content);

  /// Shorthand for an assistant message.
  const ChatMessage.assistant(String content)
      : this(role: ChatRole.assistant, content: content);

  /// Shorthand for a system instruction message.
  const ChatMessage.system(String content)
      : this(role: ChatRole.system, content: content);

  /// The role of the message author.
  final ChatRole role;

  /// The text content of the message.
  final String content;

  /// Serialises to the OpenAI / Mistral wire format.
  ///
  /// Other providers may need custom mapping (e.g. Gemini uses "model"
  /// instead of "assistant").
  Map<String, dynamic> toJson() => {
        'role': role.name,
        'content': content,
      };

  @override
  String toString() => 'ChatMessage(${role.name}: $content)';
}

/// Request payload for a chat completion call.
class ChatRequest {
  const ChatRequest({
    required this.messages,
    this.temperature,
    this.maxTokens,
    this.model,
  });

  /// Ordered list of conversation turns sent to the model.
  final List<ChatMessage> messages;

  /// Sampling temperature in the range [0, 2].
  ///
  /// Higher values produce more varied output; lower values produce more
  /// deterministic output.  Uses the provider default when `null`.
  final double? temperature;

  /// Maximum number of tokens to generate.  Uses the provider default when
  /// `null`.
  final int? maxTokens;

  /// Model identifier.  Overrides the provider-level default when set.
  final String? model;
}

/// Response from a chat completion call.
class ChatResponse {
  const ChatResponse({
    required this.message,
    this.inputTokens,
    this.outputTokens,
    this.model,
    this.finishReason,
  });

  /// The assistant message produced by the model.
  final ChatMessage message;

  /// Number of tokens consumed by the input (prompt + history).
  final int? inputTokens;

  /// Number of tokens in the generated reply.
  final int? outputTokens;

  /// Model identifier as reported by the provider.
  final String? model;

  /// Reason the generation stopped (e.g. `"stop"`, `"length"`,
  /// `"end_turn"`).
  final String? finishReason;

  @override
  String toString() =>
      'ChatResponse(model: $model, finishReason: $finishReason, '
      'tokens: ${inputTokens ?? "?"}→${outputTokens ?? "?"})';
}
