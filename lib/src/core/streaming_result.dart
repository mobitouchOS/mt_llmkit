// lib/src/core/streaming_result.dart

import 'performance_metrics.dart';

/// Streaming chunk with optional performance metrics
class StreamingChunk {
  /// Text chunk
  final String text;

  /// Current performance metrics (calculated so far)
  final PerformanceMetrics? metrics;

  /// Whether this is the last chunk
  final bool isFinal;

  StreamingChunk({required this.text, this.metrics, this.isFinal = false});

  @override
  String toString() {
    return 'StreamingChunk(text: "$text", metrics: $metrics, isFinal: $isFinal)';
  }
}
