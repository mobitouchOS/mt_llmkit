// lib/src/presentation/llm_plugin.dart

import 'dart:async';

import '../core/llm_config.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import '../data/providers/local_gguf_provider.dart';
import '../data/providers/openai_provider.dart';
import '../domain/providers/llm_provider.dart';
import '../utils/llm_utils.dart';

/// Główna klasa pluginu llmcpp.
///
/// Przyjmuje [LLMProvider] jako zależność (wzorzec Dependency Injection),
/// co umożliwia podmianę dostawcy LLM bez modyfikacji kodu klienta.
///
/// Deleguje operacje do wybranego providera i wzbogaca surowy stream tokenów
/// o live metryki wydajności ([sendPromptStream]).
///
/// ## Wzorce projektowe
/// - **Dependency Injection**: provider przekazywany przez konstruktor
/// - **Factory Method**: wygodne konstruktory `LlmPlugin.localGGUF()`, `.openAI()`
/// - **Facade**: uproszczony interfejs nad złożoną logiką providerów
/// - **Decorator**: `sendPromptStream()` dekoruje `Stream<String>` o metryki
///
/// ## Użycie
///
/// ```dart
/// // Z lokalnym modelem GGUF (Isolate, brak blokowania UI)
/// final plugin = LlmPlugin.localGGUF(
///   modelPath: '/path/to/model.gguf',
///   config: LlmConfig(temp: 0.7, nCtx: 2048),
/// );
/// await plugin.initialize();
///
/// // Streaming z live metrics
/// plugin.sendPromptStream('Napisz wiersz o Krakowie').listen((chunk) {
///   if (chunk.text.isNotEmpty) print(chunk.text);
///   if (chunk.isFinal) print('${chunk.metrics?.tokensPerSecond} t/s');
/// });
///
/// await plugin.dispose();
/// ```
class LlmPlugin {
  /// Aktywny dostawca LLM
  final LLMProvider _provider;

  /// Konfiguracja przekazywana do providera przy inicjalizacji
  final Map<String, dynamic> _initConfig;

  bool _isInitialized = false;

  /// Konstruktor z DI — przekaż dowolny [LLMProvider].
  ///
  /// Preferuj factory methods ([localGGUF], [openAI], [custom])
  /// zamiast bezpośredniego konstruktora.
  LlmPlugin(this._provider, this._initConfig);

  // ── Factory methods ────────────────────────────────────────────────────

  /// Tworzy plugin z lokalnym modelem GGUF z obsługą Isolate.
  ///
  /// Generowanie odbywa się w osobnym Isolate ([LlamaParent]),
  /// co gwarantuje brak blokowania wątku UI nawet przy dużych modelach.
  ///
  /// [modelPath] — ścieżka bezwzględna do pliku .gguf
  /// [config] — opcjonalna konfiguracja (temperatura, GPU layers, kontekst itp.)
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

  /// Tworzy plugin z OpenAI API (streaming przez SSE).
  ///
  /// **Uwaga:** [OpenAIProvider] jest obecnie szkieletem.
  /// Metody rzucają [UnimplementedError] do czasu pełnej implementacji.
  ///
  /// [apiKey] — klucz API OpenAI (format: "sk-...")
  /// [model] — nazwa modelu (domyślnie: "gpt-4o-mini")
  /// [baseUrl] — opcjonalny URL bazowy (np. Azure OpenAI endpoint)
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

  /// Tworzy plugin z niestandardowym dostawcą.
  ///
  /// Przydatne do testowania (mock providerów) i własnych integracji
  /// (Anthropic, Google Gemini, Ollama itp.).
  ///
  /// ```dart
  /// // Przykład z mockiem w testach
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

  /// Inicjalizuje plugin (ładuje model / konfiguruje API).
  ///
  /// Musi być wywołane przed użyciem [sendPrompt] lub [sendPromptStream].
  /// Kolejne wywołania są ignorowane (idempotentne).
  Future<void> initialize() async {
    if (_isInitialized) return;
    await _provider.initialize(_initConfig);
    _isInitialized = true;
  }

  /// Zwalnia wszystkie zasoby.
  ///
  /// Po wywołaniu plugin nie może być ponownie użyty.
  /// Bezpieczne do wywołania wielokrotnego.
  Future<void> dispose() async {
    if (!_isInitialized) return;
    await _provider.dispose();
    _isInitialized = false;
  }

  // ── Generation API ─────────────────────────────────────────────────────

  /// Wysyła prompt i zwraca stream surowych tokenów (String).
  ///
  /// Najprostsza metoda — bez metryk wydajności.
  /// Stream jest bezpieczny dla wątku UI.
  ///
  /// [parameters] — opcjonalne parametry specyficzne dla providera
  Stream<String> sendPrompt(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    _ensureInitialized();
    return _provider.sendPrompt(prompt, parameters: parameters);
  }

  /// Wysyła prompt i zwraca stream [StreamingChunk] z live metrics.
  ///
  /// Dekoruje `Stream<String>` z providera, wzbogacając każdy token
  /// o aktualne metryki wydajności obliczane w locie.
  ///
  /// Każdy [StreamingChunk] zawiera:
  ///   - [StreamingChunk.text] — fragment tekstu (pusty dla finalnego chunku)
  ///   - [StreamingChunk.metrics] — aktualne metryki (t/s, ms/token, łączny czas)
  ///   - [StreamingChunk.isFinal] — `true` dla ostatniego chunku ze zbiorczymi metrykami
  ///
  /// Zalecana metoda dla warstwy UI — umożliwia wyświetlanie tokenów
  /// w czasie rzeczywistym wraz z paskiem metryk wydajności.
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    Map<String, dynamic>? parameters,
  }) {
    _ensureInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    // StreamTransformer dekoruje surowy Stream<String> o metryki.
    // Obliczenia metryk są lekkie (szacowanie tokenów + datetime) —
    // mogą bezpiecznie wykonywać się na UI thread.
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
              // Finalny chunk — puste text, kompletne metryki podsumowujące
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

  // ── Gettery ────────────────────────────────────────────────────────────

  /// Czy plugin jest zainicjalizowany i gotowy do generowania
  bool get isInitialized => _isInitialized;

  /// Aktywny provider (przydatne do inspekcji w testach)
  LLMProvider get provider => _provider;

  // ── Prywatne helpers ───────────────────────────────────────────────────

  void _ensureInitialized() {
    if (!_isInitialized) {
      throw StateError(
        'LlmPlugin nie jest zainicjalizowany. Wywołaj initialize() najpierw.',
      );
    }
  }
}
