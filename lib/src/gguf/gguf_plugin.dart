// lib/src/gguf/gguf_plugin.dart

import 'package:llamadart/llamadart.dart' show LlamaImageContent;

import '../core/llm_config.dart';
import '../core/llm_interface.dart';
import '../core/streaming_result.dart';
import '../models/llm_model_base.dart';
import '../models/llm_model_isolated.dart';
import '../models/llm_model_standard.dart';

/// Backend used by [GgufPlugin].
///
/// - [isolate]: runs in a Dart Isolate — **recommended**, no UI blocking.
/// - [inProcess]: runs on the calling thread — lighter startup.
enum GGUFBackend { isolate, inProcess }

/// Unified plugin for running local GGUF models.
///
/// Implements [LlmInterface] and selects the appropriate backend internally:
/// [LlmModelIsolated] (isolate, default) or [LlmModelStandard] (in-process).
class GgufPlugin implements LlmInterface {
  final GGUFBackend backend;
  final LlmConfig config;

  LlmModelBase? _model;

  GgufPlugin({
    this.backend = GGUFBackend.isolate,
    this.config = const LlmConfig(),
  });

  @override
  Future<void> loadModel(String localPath) async {
    _model?.dispose();
    _model = backend == GGUFBackend.isolate
        ? LlmModelIsolated(config)
        : LlmModelStandard(config);
    await _model!.loadModel(localPath);
  }

  @override
  void dispose() {
    _model?.dispose();
    _model = null;
  }

  @override
  Stream<String> sendPrompt(String prompt) {
    _ensureInitialized();
    return _model!.sendPrompt(prompt);
  }

  @override
  Future<String> sendPromptComplete(String prompt) {
    _ensureInitialized();
    return _model!.sendPromptComplete(prompt);
  }

  @override
  Stream<StreamingChunk> sendPromptStream(String prompt) {
    _ensureInitialized();
    return _model!.sendPromptStream(prompt);
  }

  @override
  Stream<String> sendPromptWithImages(String prompt, List<LlamaImageContent> images) {
    _ensureInitialized();
    return _model!.sendPromptWithImages(prompt, images);
  }

  @override
  Future<String> sendPromptCompleteWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    _ensureInitialized();
    return _model!.sendPromptCompleteWithImages(prompt, images);
  }

  @override
  Stream<StreamingChunk> sendPromptStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    _ensureInitialized();
    return _model!.sendPromptStreamWithImages(prompt, images);
  }

  @override
  bool get isGenerating => _model?.isGenerating ?? false;

  @override
  bool get isInitialized => _model?.isInitialized ?? false;

  /// Resets the conversation context without reloading the model.
  ///
  /// Only supported with [GGUFBackend.inProcess].
  /// Throws [UnsupportedError] with [GGUFBackend.isolate].
  @override
  void clean() {
    _ensureInitialized();
    if (backend == GGUFBackend.inProcess) {
      _model!.clean();
    } else {
      throw UnsupportedError('clean() requires GGUFBackend.inProcess.');
    }
  }

  void _ensureInitialized() {
    if (_model == null || !_model!.isInitialized) {
      throw StateError('GgufPlugin is not initialized. Call loadModel() first.');
    }
  }
}
