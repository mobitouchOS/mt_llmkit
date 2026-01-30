// lib/src/core/generation_result.dart

import 'performance_metrics.dart';

/// Result of text generation including content and performance metrics
class GenerationResult {
  /// Generated text content
  final String text;

  /// Performance metrics for the generation
  final PerformanceMetrics? metrics;

  /// Whether the generation completed successfully
  final bool isComplete;

  /// Error message if generation failed
  final String? error;

  GenerationResult({
    required this.text,
    this.metrics,
    this.isComplete = true,
    this.error,
  });

  /// Creates a successful generation result
  factory GenerationResult.success({
    required String text,
    PerformanceMetrics? metrics,
  }) {
    return GenerationResult(text: text, metrics: metrics, isComplete: true);
  }

  /// Creates a failed generation result
  factory GenerationResult.failure({required String error, String text = ''}) {
    return GenerationResult(text: text, error: error, isComplete: false);
  }

  /// Whether the generation was successful
  bool get isSuccess => isComplete && error == null;

  @override
  String toString() {
    if (isSuccess) {
      return 'GenerationResult(text length: ${text.length}, metrics: $metrics)';
    } else {
      return 'GenerationResult(error: $error)';
    }
  }

  /// Converts to JSON-compatible map
  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'metrics': metrics?.toJson(),
      'isComplete': isComplete,
      'error': error,
    };
  }

  /// Creates from JSON map
  factory GenerationResult.fromJson(Map<String, dynamic> json) {
    return GenerationResult(
      text: json['text'] as String,
      metrics: json['metrics'] != null
          ? PerformanceMetrics.fromJson(json['metrics'] as Map<String, dynamic>)
          : null,
      isComplete: json['isComplete'] as bool,
      error: json['error'] as String?,
    );
  }
}
