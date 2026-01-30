// lib/src/core/performance_metrics.dart

/// Performance metrics for LLM text generation
class PerformanceMetrics {
  /// Number of tokens generated
  final int tokensGenerated;

  /// Total time taken for generation in milliseconds
  final int durationMs;

  /// Tokens per second (t/s)
  final double tokensPerSecond;

  /// Average time per token in milliseconds
  final double msPerToken;

  /// Start timestamp
  final DateTime startTime;

  /// End timestamp
  final DateTime endTime;

  PerformanceMetrics({
    required this.tokensGenerated,
    required this.durationMs,
    required this.tokensPerSecond,
    required this.msPerToken,
    required this.startTime,
    required this.endTime,
  });

  /// Creates performance metrics from generation data
  factory PerformanceMetrics.fromGeneration({
    required int tokenCount,
    required DateTime startTime,
    required DateTime endTime,
  }) {
    final duration = endTime.difference(startTime);
    final durationMs = duration.inMilliseconds;
    final tokensPerSecond = durationMs > 0
        ? (tokenCount * 1000) / durationMs
        : 0.0;
    final msPerToken = tokenCount > 0 ? durationMs / tokenCount : 0.0;

    return PerformanceMetrics(
      tokensGenerated: tokenCount,
      durationMs: durationMs,
      tokensPerSecond: tokensPerSecond,
      msPerToken: msPerToken,
      startTime: startTime,
      endTime: endTime,
    );
  }

  /// Duration of generation
  Duration get duration => Duration(milliseconds: durationMs);

  @override
  String toString() {
    return 'PerformanceMetrics('
        'tokens: $tokensGenerated, '
        'duration: ${durationMs}ms, '
        't/s: ${tokensPerSecond.toStringAsFixed(2)}, '
        'ms/token: ${msPerToken.toStringAsFixed(2)}'
        ')';
  }

  /// Converts to JSON-compatible map
  Map<String, dynamic> toJson() {
    return {
      'tokensGenerated': tokensGenerated,
      'durationMs': durationMs,
      'tokensPerSecond': tokensPerSecond,
      'msPerToken': msPerToken,
      'startTime': startTime.toIso8601String(),
      'endTime': endTime.toIso8601String(),
    };
  }

  /// Creates from JSON map
  factory PerformanceMetrics.fromJson(Map<String, dynamic> json) {
    return PerformanceMetrics(
      tokensGenerated: json['tokensGenerated'] as int,
      durationMs: json['durationMs'] as int,
      tokensPerSecond: (json['tokensPerSecond'] as num).toDouble(),
      msPerToken: (json['msPerToken'] as num).toDouble(),
      startTime: DateTime.parse(json['startTime'] as String),
      endTime: DateTime.parse(json['endTime'] as String),
    );
  }
}
