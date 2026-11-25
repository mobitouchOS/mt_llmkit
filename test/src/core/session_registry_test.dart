import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:llamadart/llamadart.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/core/generation_event.dart';
import 'package:mt_llmkit/src/core/session_ops.dart';

/// Stands in for llamadart's ChatSession, so every rule in [SessionRegistry] —
/// trim reporting, cancellation repair, tool-result fan-out, serialization —
/// is testable without an engine, an isolate or a model file.
class FakeChatSession implements ChatSessionLike {
  @override
  String? systemPrompt;
  @override
  int? maxContextTokens;
  @override
  bool lastRequestFitContext = true;

  final List<LlamaChatMessage> _history = [];

  /// Chunks the next `create` yields.
  List<LlamaCompletionChunk> script = const [];

  /// Fires before the first chunk, standing in for `_enforceContextLimit`.
  void Function(FakeChatSession session)? onCreate;

  /// Held open so a test can control when a turn finishes.
  Completer<void>? gate;

  final List<bool> continuationFlags = [];

  @override
  List<LlamaChatMessage> get history => List.unmodifiable(_history);

  @override
  void addMessage(LlamaChatMessage message) => _history.add(message);

  @override
  void reset({bool keepSystemPrompt = true}) {
    _history.clear();
    if (!keepSystemPrompt) systemPrompt = null;
  }

  /// Mimics llamadart deleting the oldest turns to make room.
  void dropOldest(int count) => _history.removeRange(0, count);

  @override
  Stream<LlamaCompletionChunk> create(
    List<LlamaContentPart> parts, {
    GenerationParams? params,
    List<ToolDefinition>? tools,
    ToolChoice? toolChoice,
    bool parallelToolCalls = false,
    bool enableThinking = true,
    bool continuesPreviousTurn = false,
  }) async* {
    continuationFlags.add(continuesPreviousTurn);

    if (parts.isNotEmpty) {
      _history.add(
        LlamaChatMessage.withContent(
          role: LlamaChatRole.user,
          content: parts,
          continuesPreviousTurn: continuesPreviousTurn,
        ),
      );
    }

    onCreate?.call(this);

    final content = StringBuffer();
    for (final chunk in script) {
      if (gate != null) await gate!.future;
      content.write(chunk.choices.firstOrNull?.delta.content ?? '');
      yield chunk;
    }

    // ChatSession appends the assistant turn only after a clean finish.
    _history.add(
      LlamaChatMessage.withContent(
        role: LlamaChatRole.assistant,
        content: [LlamaTextContent(content.toString())],
      ),
    );
  }
}

LlamaCompletionChunk _chunk({String? text, String? thinking}) =>
    LlamaCompletionChunk(
      id: '',
      object: '',
      created: 0,
      model: '',
      choices: [
        LlamaCompletionChunkChoice(
          index: 0,
          delta: LlamaCompletionChunkDelta(content: text, thinking: thinking),
        ),
      ],
    );

({SessionRegistry registry, FakeChatSession session}) _build({
  SessionOptions options = const SessionOptions(),
  List<LlamaCompletionChunk> script = const [],
}) {
  final session = FakeChatSession()..script = script;
  var queue = Future<void>.value();

  final registry = SessionRegistry(
    createSession: (_) => session,
    baseParams: () => const GenerationParams(),
    enableThinkingDefault: () => true,
    readPerf: () async => null,
    serialize: (action) => queue = queue.then((_) => action()),
  );
  registry.open('s0', options);

  return (registry: registry, session: session);
}

Future<List<GenerationEvent>> _drain(Stream<GenerationEvent> stream) =>
    stream.toList();

void main() {
  group('turns', () {
    test('streams text and reports a clean finish', () async {
      final f = _build(
        script: [
          _chunk(text: 'Hello '),
          _chunk(text: 'there'),
        ],
      );

      final events = await _drain(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
      );

      expect(events.where((e) => !e.isFinal).map((e) => e.text), [
        'Hello ',
        'there',
      ]);
      expect(events.last.isFinal, isTrue);
      expect(events.last.fitContext, isTrue);
      expect(events.last.dropped, isEmpty);
    });

    test('keeps reasoning on its own channel', () async {
      final f = _build(
        script: [
          _chunk(thinking: 'hmm'),
          _chunk(text: 'answer'),
        ],
      );

      final events = await _drain(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
      );

      expect(events.first.thinking, 'hmm');
      expect(events.first.text, isEmpty);
      expect(events[1].text, 'answer');
    });

    test('a turn on a closed conversation errors instead of hanging', () async {
      final f = _build();
      f.registry.close('s0');

      await expectLater(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
        emitsError(isStateError),
      );
    });
  });

  group('tool results', () {
    test('become one message each, in order', () async {
      // llamadart's message JSON keeps only the first tool result per message,
      // so batching them into one would silently drop all but one.
      final f = _build(script: [_chunk(text: 'ok')]);

      await _drain(
        f.registry.turn(
          's0',
          const TurnRequest(
            toolResults: [
              ToolResult(name: 'weather', result: 'sunny'),
              ToolResult(name: 'time', result: '12:00'),
            ],
          ),
        ),
      );

      final toolMessages = f.session.history
          .where((m) => m.role == LlamaChatRole.tool)
          .toList();
      expect(toolMessages, hasLength(2));
    });

    test('continue the previous turn rather than starting a new one', () async {
      final f = _build(script: [_chunk(text: 'ok')]);

      await _drain(
        f.registry.turn(
          's0',
          const TurnRequest(
            toolResults: [ToolResult(name: 'weather', result: 'sunny')],
          ),
        ),
      );

      // Context trimming treats a turn and its tool round trips as one unit.
      expect(f.session.continuationFlags.single, isTrue);
    });
  });

  group('context trimming', () {
    test('reports exactly the turns llamadart deleted', () async {
      final f = _build(script: [_chunk(text: 'ok')]);
      f.registry.restore('s0', const [
        LlmChatMessage.user('oldest'),
        LlmChatMessage.assistant('older reply'),
        LlmChatMessage.user('recent'),
      ]);
      f.session.onCreate = (s) => s.dropOldest(2);

      final events = await _drain(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
      );

      expect(events.last.dropped.map((m) => m.text), ['oldest', 'older reply']);
    });

    test(
      'reports fitContext false when even trimming was not enough',
      () async {
        final f = _build(script: [_chunk(text: 'ok')]);
        f.session.lastRequestFitContext = false;

        final events = await _drain(
          f.registry.turn('s0', const TurnRequest(message: 'hi')),
        );

        expect(events.last.fitContext, isFalse);
      },
    );

    test('fail policy aborts before any output is consumed', () async {
      final f = _build(
        options: const SessionOptions(
          overflowPolicy: ContextOverflowPolicy.fail,
        ),
        script: [_chunk(text: 'answer built on a truncated prompt')],
      );
      f.session.lastRequestFitContext = false;

      final events = <GenerationEvent>[];
      await expectLater(
        f.registry.turn('s0', const TurnRequest(message: 'hi')).map((e) {
          events.add(e);
          return e;
        }),
        emitsError(isA<LlmContextOverflowException>()),
      );
      expect(events, isEmpty);
    });

    test('allow policy hands the answer over anyway', () async {
      final f = _build(
        options: const SessionOptions(
          overflowPolicy: ContextOverflowPolicy.allow,
        ),
        script: [_chunk(text: 'answer')],
      );
      f.session.lastRequestFitContext = false;

      final events = await _drain(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
      );

      expect(events.first.text, 'answer');
      expect(events.last.fitContext, isFalse);
    });

    test('does not guess when history identities stop lining up', () async {
      final f = _build(script: [_chunk(text: 'ok')]);
      f.registry.restore('s0', const [LlmChatMessage.user('old')]);
      // A future llamadart rewriting history instead of trimming it must not
      // read as "the whole conversation was deleted".
      f.session.onCreate = (s) {
        final rewritten = s.history
            .map((m) => LlamaChatMessage.fromText(role: m.role, text: 'x'))
            .toList();
        s.reset();
        rewritten.forEach(s.addMessage);
      };

      final events = await _drain(
        f.registry.turn('s0', const TurnRequest(message: 'hi')),
      );

      expect(events.last.dropped, isEmpty);
    });
  });

  group('history repair', () {
    test('cancelling mid-turn still leaves an assistant reply', () async {
      // ChatSession appends its assistant message only after a clean finish,
      // so without repair the next turn renders two user messages in a row.
      final f = _build(
        script: [
          _chunk(text: 'par'),
          _chunk(text: 'tial'),
        ],
      );
      f.session.gate = Completer<void>();

      final sub = f.registry
          .turn('s0', const TurnRequest(message: 'hi'))
          .listen(null);
      f.session.gate!.complete();
      await Future<void>.delayed(Duration.zero);
      await sub.cancel();
      await Future<void>.delayed(const Duration(milliseconds: 10));

      expect(f.session.history.last.role, LlamaChatRole.assistant);
      expect(
        f.session.history.where((m) => m.role == LlamaChatRole.user),
        hasLength(1),
      );
    });
  });

  group('reasoning in history', () {
    test('is stripped by default', () async {
      final f = _build(script: [_chunk(text: 'answer')]);
      f.registry.restore('s0', const [
        LlmChatMessage.assistant('older', thinking: 'previous reasoning'),
      ]);

      await _drain(f.registry.turn('s0', const TurnRequest(message: 'hi')));

      final history = f.registry.history('s0');
      expect(history.every((m) => m.thinking == null), isTrue);
    });

    test('is kept when the conversation asked for it', () async {
      final f = _build(
        options: const SessionOptions(keepThinkingInHistory: true),
        script: [_chunk(text: 'answer')],
      );
      f.registry.restore('s0', const [
        LlmChatMessage.assistant('older', thinking: 'previous reasoning'),
      ]);

      await _drain(f.registry.turn('s0', const TurnRequest(message: 'hi')));

      expect(
        f.registry.history('s0').any((m) => m.thinking == 'previous reasoning'),
        isTrue,
      );
    });
  });

  group('history transfer', () {
    test('a system message is routed to the system prompt, not dropped', () {
      // llamadart silently filters system messages out of the rendered prompt;
      // only the systemPrompt field reaches the model.
      final f = _build();

      f.registry.restore('s0', const [
        LlmChatMessage(role: LlmChatRole.system, text: 'Be terse.'),
        LlmChatMessage.user('hi'),
      ]);

      expect(f.session.systemPrompt, 'Be terse.');
      expect(f.registry.history('s0'), hasLength(1));
    });

    test('round-trips text, reasoning and tool calls', () {
      final f = _build();

      f.registry.restore('s0', const [
        LlmChatMessage.user('question'),
        LlmChatMessage.assistant(
          'reply',
          thinking: 'reasoning',
          toolCalls: [
            LlmToolCall(index: 0, id: 'c1', name: 'search', arguments: '{}'),
          ],
        ),
      ]);

      final history = f.registry.history('s0');
      expect(history[0].text, 'question');
      expect(history[1].thinking, 'reasoning');
      expect(history[1].toolCalls.single.name, 'search');
    });
  });

  group('serialization', () {
    test('turns queue instead of interleaving', () async {
      // llama.cpp chat generation runs on one context, so two turns must not
      // overlap.
      final f = _build(script: [_chunk(text: 'a')]);
      f.session.gate = Completer<void>();

      final first = _drain(
        f.registry.turn('s0', const TurnRequest(message: 'one')),
      );
      final second = _drain(
        f.registry.turn('s0', const TurnRequest(message: 'two')),
      );

      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(f.session.continuationFlags, hasLength(1)); // second not started

      f.session.gate!.complete();
      await first;
      await second;
      expect(f.session.continuationFlags, hasLength(2));
    });

    // Interrupting a turn used to wedge the queue for the rest of the
    // process: a cancelled subscription never delivers `onDone`, so the turn
    // sat on `await finished.future` forever while holding the queue, and
    // every later turn — and every load/unload — waited behind it with no
    // token, no error and no end. In the app that showed as a chat that
    // answered once and then hung on "thinking" until it was restarted.
    group('an interrupted turn frees the queue', () {
      /// Starts a turn and blocks it inside `create`, so the interruption
      /// under test lands while the turn genuinely holds the queue.
      ({Future<void> Function() interruptViaCancel, FakeChatSession session})
      gatedTurn(SessionRegistry registry, FakeChatSession session) {
        session.gate = Completer<void>();
        final sub = registry
            .turn('s0', const TurnRequest(message: 'hi'))
            .listen(null);
        return (interruptViaCancel: sub.cancel, session: session);
      }

      /// Runs a fresh turn and fails the test if the queue never gets to it.
      Future<List<GenerationEvent>> nextTurn(SessionRegistry registry) {
        registry.open('s1', const SessionOptions());
        return _drain(
          registry.turn('s1', const TurnRequest(message: 'again')),
        ).timeout(
          const Duration(seconds: 5),
          onTimeout: () => fail('queue never ran the next turn'),
        );
      }

      test('after the listener cancels', () async {
        final f = _build(script: [_chunk(text: 'partial')]);
        final turn = gatedTurn(f.registry, f.session);
        await Future<void>.delayed(Duration.zero);

        await turn.interruptViaCancel();

        // Let the abandoned turn run to completion had it been resumed, so a
        // pass cannot come from the gate being released behind our back.
        f.session.gate = null;

        expect((await nextTurn(f.registry)).last.isFinal, isTrue);
      });

      test('after the conversation is closed mid-turn', () async {
        // The path the app actually takes: its generation stream has a
        // `finally { conversation.close(); }`, and close() is dispatched
        // outside the queue, so it lands while the turn is still decoding.
        final f = _build(script: [_chunk(text: 'partial')]);
        gatedTurn(f.registry, f.session);
        await Future<void>.delayed(Duration.zero);

        f.registry.close('s0');
        f.session.gate = null;

        expect((await nextTurn(f.registry)).last.isFinal, isTrue);
      });
    });
  });
}
