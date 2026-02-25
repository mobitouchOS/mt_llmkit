// lib/src/gguf/gguf_plugin.dart

import '../core/llm_config.dart';
import '../core/llm_interface.dart';
import '../core/streaming_result.dart';
import '../models/llm_model_base.dart';
import '../models/llm_model_isolated.dart';
import '../models/llm_model_standard.dart';

/// Backend used by [GgufPlugin].
///
/// - [isolate]: runs in a Dart Isolate via [LlamaParent] — **recommended**,
///   no UI blocking. Required when running multiple models simultaneously
///   (e.g. RAG with embedding + generation model).
/// - [inProcess]: runs on the calling thread via [Llama] — lighter startup,
///   supports [GgufPlugin.clean] to reset context without reloading the model.
enum GGUFBackend { isolate, inProcess }

/// Unified plugin for running local GGUF models.
///
/// Implements [LlmInterface] and selects the appropriate backend internally:
/// [LlmModelIsolated] (isolate, default) or [LlmModelStandard] (in-process).
///
/// ## Usage
///
/// ```dart
/// final plugin = GgufPlugin(
///   backend: GGUFBackend.isolate,
///   config: LlmConfig(temp: 0.7, nCtx: 2048),
/// );
/// await plugin.loadModel('/path/to/model.gguf');
///
/// // Streaming with live metrics
/// plugin.sendPromptStream('Hello').listen((chunk) {
///   if (chunk.text.isNotEmpty) print(chunk.text);
///   if (chunk.isFinal) print('Done. t/s: ${chunk.metrics?.tokensPerSecond}');
/// });
///
/// // Blocking — full response as String
/// final response = await plugin.sendPromptComplete('Hello');
///
/// // Check if generation is in progress
/// print(plugin.isGenerating);
///
/// plugin.dispose();
/// ```
///
/// ## clean() — context reset without model reload
///
/// Only supported with [GGUFBackend.inProcess]. Throws [UnsupportedError]
/// when used with [GGUFBackend.isolate].
class GgufPlugin implements LlmInterface {
  /// Backend used for inference.
  final GGUFBackend backend;

  /// Configuration (temperature, GPU layers, context size, etc.).
  final LlmConfig config;

  LlmModelBase? _model;

  GgufPlugin({
    this.backend = GGUFBackend.isolate,
    this.config = const LlmConfig(),
  });

  // ── Lifecycle ──────────────────────────────────────────────────────────

  /// Loads the GGUF model from [localPath].
  ///
  /// Must be called before [sendPrompt], [sendPromptComplete], or
  /// [sendPromptStream]. Subsequent calls reload the model.
  ///
  /// Throws [FileSystemException] if the file does not exist.
  /// Throws [StateError] if called after [dispose].
  @override
  Future<void> loadModel(String localPath) async {
    _model?.dispose();
    _model = backend == GGUFBackend.isolate
        ? LlmModelIsolated(config)
        : LlmModelStandard(config);
    await _model!.loadModel(localPath);
  }

  /// Releases all resources (Isolate / FFI model memory).
  ///
  /// Safe to call multiple times.
  @override
  void dispose() {
    _model?.dispose();
    _model = null;
  }

  // ── Generation ─────────────────────────────────────────────────────────

  /// Sends a prompt and returns a stream of raw tokens.
  ///
  /// Use [isGenerating] to check if generation is still in progress.
  ///
  /// Throws [StateError] if the plugin is not initialized.
  @override
  Stream<String> sendPrompt(String prompt) {
    _ensureInitialized();
    return _model!.sendPrompt(prompt);
  }

  /// Sends a prompt and returns the complete response as a [String].
  ///
  /// Blocks until generation is finished.
  ///
  /// Throws [StateError] if the plugin is not initialized.
  @override
  Future<String> sendPromptComplete(String prompt) {
    _ensureInitialized();
    return _model!.sendPromptComplete(prompt);
  }

  /// Sends a prompt and returns a stream of [StreamingChunk] with live
  /// performance metrics.
  ///
  /// Each chunk carries [StreamingChunk.text], [StreamingChunk.metrics], and
  /// [StreamingChunk.isFinal] to detect the end of generation.
  ///
  /// Throws [StateError] if the plugin is not initialized.
  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) {
    _ensureInitialized();
    return _model!.sendPromptStream(prompt);
  }

  // ── Status ─────────────────────────────────────────────────────────────

  /// Whether generation is currently in progress.
  @override
  bool get isGenerating => _model?.isGenerating ?? false;

  /// Whether the plugin has a loaded model and is ready for generation.
  @override
  bool get isInitialized => _model?.isInitialized ?? false;

  // ── Context reset ──────────────────────────────────────────────────────

  /// Resets the conversation context without reloading the model.
  ///
  /// Only supported with [GGUFBackend.inProcess].
  /// Throws [UnsupportedError] with [GGUFBackend.isolate].
  /// Throws [StateError] if the plugin is not initialized.
  @override
  void clean() {
    _ensureInitialized();
    if (backend == GGUFBackend.inProcess) {
      _model!.clean();
    } else {
      throw UnsupportedError('clean() requires GGUFBackend.inProcess.');
    }
  }

  // ── Private ────────────────────────────────────────────────────────────

  void _ensureInitialized() {
    if (_model == null || !_model!.isInitialized) {
      throw StateError(
        'GgufPlugin is not initialized. Call loadModel() first.',
      );
    }
  }
}
