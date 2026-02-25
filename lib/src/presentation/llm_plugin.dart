// lib/src/presentation/llm_plugin.dart

import 'dart:async';

import '../core/llm_config.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import '../data/providers/local_gguf_provider.dart';
import '../data/providers/openai_provider.dart';
import '../domain/providers/llm_provider.dart';
import '../utils/llm_utils.dart';

/// Main class of the llmcpp plugin.
///
/// Accepts [LLMProvider] as a dependency (Dependency Injection pattern),
/// allowing the LLM provider to be swapped without modifying client code.
///
/// Delegates operations to the selected provider and enriches the raw token
/// stream with live performance metrics ([sendPromptStream]).
///
/// ## Design patterns
/// - **Dependency Injection**: provider passed via constructor
/// - **Factory Method**: convenience constructors `LlmPlugin.localGGUF()`, `.openAI()`
/// - **Facade**: simplified interface over complex provider logic
/// - **Decorator**: `sendPromptStream()` decorates `Stream<String>` with metrics
///
/// ## Usage
///
/// ```dart
/// // With a local GGUF model (Isolate, no UI blocking)
/// final plugin = LlmPlugin.localGGUF(
///   modelPath: '/path/to/model.gguf',
///   config: LlmConfig(temp: 0.7, nCtx: 2048),
/// );
/// await plugin.initialize();
///
/// // Streaming with live metrics
/// plugin.sendPromptStream('Write a poem about Krakow').listen((chunk) {
///   if (chunk.text.isNotEmpty) print(chunk.text);
///   if (chunk.isFinal) print('${chunk.metrics?.tokensPerSecond} t/s');
/// });
///
/// await plugin.dispose();
/// ```
class LlmPlugin {
  /// Active LLM provider
  final LLMProvider _provider;

  /// Configuration passed to the provider during initialization
  final Map<String, dynamic> _initConfig;

  bool _isInitialized = false;

  /// DI constructor — pass any [LLMProvider].
  ///
  /// Prefer factory methods ([localGGUF], [openAI], [custom])
  /// over using this constructor directly.
  LlmPlugin(this._provider, this._initConfig);

  // ── Factory methods ────────────────────────────────────────────────────

  /// Creates a plugin with a local GGUF model backed by an Isolate.
  ///
  /// Generation runs in a separate Isolate ([LlamaParent]),
  /// guaranteeing no UI thread blocking even with large models.
  ///
  /// [modelPath] — absolute path to the .gguf file
  /// [config] — optional configuration (temperature, GPU layers, context, etc.)
  factory LlmPlugin.localGGUF({
    required String modelPath,
    LlmConfig? config,
  }) {
    return LlmPlugin(
      LocalGGUFProvider(),
      {
        'modelPath': modelPath,
        if (config != null) 'llmConfig': config,
      },
    );
  }

  /// Creates a plugin with the OpenAI API (streaming via SSE).
  ///
  /// **Note:** [OpenAIProvider] is currently a skeleton.
  /// Methods throw [UnimplementedError] until fully implemented.
  ///
  /// [apiKey] — OpenAI API key (format: "sk-...")
  /// [model] — model name (default: "gpt-4o-mini")
  /// [baseUrl] — optional base URL (e.g. Azure OpenAI endpoint)
  factory LlmPlugin.openAI({
    required String apiKey,
    String model = 'gpt-4o-mini',
    String? baseUrl,
  }) {
    return LlmPlugin(
      OpenAIProvider(),
      {
        'apiKey': apiKey,
        'model': model,
        if (baseUrl != null) 'baseUrl': baseUrl,
      },
    );
  }

  /// Creates a plugin with a custom provider.
  ///
  /// Useful for testing (mock providers) and custom integrations
  /// (Anthropic, Google Gemini, Ollama, etc.).
  ///
  /// ```dart
  /// // Example with a mock in tests
  /// final plugin = LlmPlugin.custom(
  ///   provider: MockLLMProvider(),
  ///   config: {'modelPath': 'fake.gguf'},
  /// );
  /// ```
  factory LlmPlugin.custom({
    required LLMProvider provider,
    required Map<String, dynamic> config,
  }) {
    return LlmPlugin(provider, config);
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────

  /// Initializes the plugin (loads model / configures API).
  ///
  /// Must be called before using [sendPrompt] or [sendPromptStream].
  /// Subsequent calls are ignored (idempotent).
  Future<void> initialize() async {
    if (_isInitialized) return;
    await _provider.initialize(_initConfig);
    _isInitialized = true;
  }

  /// Releases all resources.
  ///
  /// After calling this, the plugin cannot be used again.
  /// Safe to call multiple times.
  Future<void> dispose() async {
    if (!_isInitialized) return;
    await _provider.dispose();
    _isInitialized = false;
  }

  // ── Generation API ─────────────────────────────────────────────────────

  /// Sends a prompt and returns a stream of raw tokens (String).
  ///
  /// Simplest method — no performance metrics.
  /// The stream is UI-thread-safe.
  ///
  /// [parameters] — optional provider-specific parameters
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    _ensureInitialized();
    return _provider.sendPrompt(prompt, parameters: parameters);
  }

  /// Sends a prompt and returns a stream of [StreamingChunk] with live metrics.
  ///
  /// Decorates the `Stream<String>` from the provider, enriching each token
  /// with current performance metrics computed on the fly.
  ///
  /// Each [StreamingChunk] contains:
  ///   - [StreamingChunk.text] — text fragment (empty for the final chunk)
  ///   - [StreamingChunk.metrics] — current metrics (t/s, ms/token, total time)
  ///   - [StreamingChunk.isFinal] — `true` for the last chunk with aggregate metrics
  ///
  /// Recommended method for the UI layer — allows displaying tokens
  /// in real time alongside a performance metrics bar.
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    _ensureInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    // StreamTransformer decorates the raw Stream<String> with metrics.
    // Metric computations are lightweight (token estimation + datetime) —
    // they can safely run on the UI thread.
    return _provider
        .sendPrompt(prompt, parameters: parameters)
        .transform(
          StreamTransformer<String, StreamingChunk>.fromHandlers(
            handleData: (token, sink) {
              totalTokenCount += LlmUtils.estimateTokenCount(token);

              final metrics = PerformanceMetrics.fromGeneration(
                tokenCount: totalTokenCount,
                startTime: startTime,
                endTime: DateTime.now(),
              );

              sink.add(StreamingChunk(
                text: token,
                metrics: metrics,
                isFinal: false,
              ));
            },
            handleDone: (sink) {
              // Final chunk — empty text, complete summary metrics
              final finalMetrics = PerformanceMetrics.fromGeneration(
                tokenCount: totalTokenCount,
                startTime: startTime,
                endTime: DateTime.now(),
              );

              sink.add(StreamingChunk(
                text: '',
                metrics: finalMetrics,
                isFinal: true,
              ));
              sink.close();
            },
            handleError: (error, stackTrace, sink) {
              sink.addError(error, stackTrace);
            },
          ),
        );
  }

  // ── Getters ────────────────────────────────────────────────────────────

  /// Whether the plugin is initialized and ready for generation
  bool get isInitialized => _isInitialized;

  /// Active provider (useful for inspection in tests)
  LLMProvider get provider => _provider;

  // ── Private helpers ────────────────────────────────────────────────────

  void _ensureInitialized() {
    if (!_isInitialized) {
      throw StateError(
        'LlmPlugin is not initialized. Call initialize() first.',
      );
    }
  }
}
