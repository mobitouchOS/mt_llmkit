// lib/src/models/llm_model_base.dart
import 'package:flutter/foundation.dart';

import '../core/llm_interface.dart';

abstract class LlmModelBase implements LlmInterface {
  bool _isInitialized = false;
  bool _isDisposed = false;

  bool get isInitialized => _isInitialized;
  bool get isDisposed => _isDisposed;

  @protected
  void markAsInitialized() {
    _isInitialized = true;
  }

  @protected
  void markAsDisposed() {
    _isDisposed = true;
    _isInitialized = false;
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
}
