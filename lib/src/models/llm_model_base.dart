// lib/src/models/llm_model_base.dart
import 'package:flutter/foundation.dart';
import 'package:llamadart/llamadart.dart' show LlamaImageContent;

import '../core/llm_interface.dart';
import '../core/streaming_result.dart';

abstract class LlmModelBase implements LlmInterface {
  bool _isInitialized = false;
  bool _isDisposed = false;
  bool _isGenerating = false;

  @override
  bool get isInitialized => _isInitialized;
  bool get isDisposed => _isDisposed;

  @override
  bool get isGenerating => _isGenerating;

  @protected
  void markAsInitialized() {
    _isInitialized = true;
  }

  @protected
  void markAsDisposed() {
    _isDisposed = true;
    _isInitialized = false;
    _isGenerating = false;
  }

  @protected
  void markGenerationStart() {
    _isGenerating = true;
  }

  @protected
  void markGenerationEnd() {
    _isGenerating = false;
  }

  @protected
  void checkInitialized() {
    if (!_isInitialized) {
      throw StateError('Model not initialized. Call loadModel() first.');
    }
    if (_isDisposed) {
      throw StateError('Model has been disposed.');
    }
  }

  @protected
  void checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Model has been disposed.');
    }
  }

  @override
  Stream<String> sendPromptWithImages(String prompt, List<LlamaImageContent> images) {
    throw UnsupportedError(
      'Vision not supported. Set mmprojPath in LlmConfig to enable it.',
    );
  }

  @override
  Future<String> sendPromptCompleteWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    throw UnsupportedError(
      'Vision not supported. Set mmprojPath in LlmConfig to enable it.',
    );
  }

  @override
  Stream<StreamingChunk> sendPromptStreamWithImages(
    String prompt,
    List<LlamaImageContent> images,
  ) {
    throw UnsupportedError(
      'Vision not supported. Set mmprojPath in LlmConfig to enable it.',
    );
  }
}
