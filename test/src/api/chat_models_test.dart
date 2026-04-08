// test/src/api/chat_models_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/src/api/chat_models.dart';

void main() {
  group('ChatMessage', () {
    test('user() convenience constructor sets role and content', () {
      const msg = ChatMessage.user('Hello');
      expect(msg.role, ChatRole.user);
      expect(msg.content, 'Hello');
    });

    test('assistant() convenience constructor', () {
      const msg = ChatMessage.assistant('Hi there');
      expect(msg.role, ChatRole.assistant);
      expect(msg.content, 'Hi there');
    });

    test('system() convenience constructor', () {
      const msg = ChatMessage.system('Be helpful');
      expect(msg.role, ChatRole.system);
      expect(msg.content, 'Be helpful');
    });

    test('toJson() serialises role as its name', () {
      const msg = ChatMessage.user('Test');
      expect(msg.toJson(), {'role': 'user', 'content': 'Test'});
    });

    test('toJson() uses "assistant" for assistant role', () {
      const msg = ChatMessage.assistant('Reply');
      expect(msg.toJson()['role'], 'assistant');
    });

    test('toJson() uses "system" for system role', () {
      const msg = ChatMessage.system('Instruction');
      expect(msg.toJson()['role'], 'system');
    });

    test('toString() contains role and content', () {
      const msg = ChatMessage.user('Hello');
      expect(msg.toString(), contains('user'));
      expect(msg.toString(), contains('Hello'));
    });
  });

  group('ChatRequest', () {
    test('stores messages and all optional parameters', () {
      final req = ChatRequest(
        messages: [ChatMessage.user('Hi')],
        temperature: 0.7,
        maxTokens: 512,
        model: 'gpt-4o',
      );
      expect(req.messages.length, 1);
      expect(req.temperature, 0.7);
      expect(req.maxTokens, 512);
      expect(req.model, 'gpt-4o');
    });

    test('optional fields default to null', () {
      final req = ChatRequest(messages: [ChatMessage.user('Hi')]);
      expect(req.temperature, isNull);
      expect(req.maxTokens, isNull);
      expect(req.model, isNull);
    });

    test('accepts multiple messages preserving order', () {
      final messages = [
        ChatMessage.system('Be concise.'),
        ChatMessage.user('Hello'),
        ChatMessage.assistant('Hi!'),
        ChatMessage.user('Bye'),
      ];
      final req = ChatRequest(messages: messages);
      expect(req.messages.map((m) => m.role).toList(), [
        ChatRole.system,
        ChatRole.user,
        ChatRole.assistant,
        ChatRole.user,
      ]);
    });
  });

  group('ChatResponse', () {
    test('stores message and all metadata', () {
      final response = ChatResponse(
        message: ChatMessage.assistant('Hello!'),
        inputTokens: 10,
        outputTokens: 5,
        model: 'gpt-4o-mini',
        finishReason: 'stop',
      );
      expect(response.message.role, ChatRole.assistant);
      expect(response.message.content, 'Hello!');
      expect(response.inputTokens, 10);
      expect(response.outputTokens, 5);
      expect(response.model, 'gpt-4o-mini');
      expect(response.finishReason, 'stop');
    });

    test('optional fields default to null', () {
      final response = ChatResponse(message: ChatMessage.assistant('Hi'));
      expect(response.inputTokens, isNull);
      expect(response.outputTokens, isNull);
      expect(response.model, isNull);
      expect(response.finishReason, isNull);
    });

    test('toString() contains model and finishReason', () {
      final response = ChatResponse(
        message: ChatMessage.assistant('Hi'),
        model: 'gpt-4o-mini',
        finishReason: 'stop',
      );
      expect(response.toString(), contains('gpt-4o-mini'));
      expect(response.toString(), contains('stop'));
    });
  });

  group('ChatRole', () {
    test('has user, assistant, and system values', () {
      expect(ChatRole.values, contains(ChatRole.user));
      expect(ChatRole.values, contains(ChatRole.assistant));
      expect(ChatRole.values, contains(ChatRole.system));
      expect(ChatRole.values.length, 3);
    });

    test('name property returns lowercase string', () {
      expect(ChatRole.user.name, 'user');
      expect(ChatRole.assistant.name, 'assistant');
      expect(ChatRole.system.name, 'system');
    });
  });
}
