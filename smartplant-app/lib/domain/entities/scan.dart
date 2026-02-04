import 'package:equatable/equatable.dart';
import 'leaf_result.dart';

/// Scan entity representing a complete plant analysis
class Scan extends Equatable {
  final int id;
  final DateTime timestamp;
  final String plantType;
  final String status;
  final ScanSummary summary;
  final List<LeafResult> leaves;
  final Interpretation interpretation;

  const Scan({
    required this.id,
    required this.timestamp,
    required this.plantType,
    required this.status,
    required this.summary,
    required this.leaves,
    required this.interpretation,
  });

  @override
  List<Object?> get props => [
        id,
        timestamp,
        plantType,
        status,
        summary,
        leaves,
        interpretation,
      ];
}

/// Summary of the plant analysis
class ScanSummary extends Equatable {
  final double healthScore;
  final String condition;
  final String predictedClass;
  final double confidence;
  final Map<String, double> classProbabilities;
  final double avgLesionAreaPercent;
  final int totalLesionCount;

  const ScanSummary({
    required this.healthScore,
    required this.condition,
    required this.predictedClass,
    required this.confidence,
    required this.classProbabilities,
    required this.avgLesionAreaPercent,
    required this.totalLesionCount,
  });

  /// Get confidence as percentage (0-100)
  double get confidencePercent => confidence * 100;

  /// Check if prediction is healthy
  bool get isHealthy => condition.toLowerCase() == 'healthy';

  /// Check if prediction is uncertain
  bool get isUncertain => condition.toLowerCase() == 'uncertain';

  @override
  List<Object?> get props => [
        healthScore,
        condition,
        predictedClass,
        confidence,
        classProbabilities,
        avgLesionAreaPercent,
        totalLesionCount,
      ];
}

/// Interpretation notes from the API
class Interpretation extends Equatable {
  final String classificationNote;
  final String healthScoreNote;
  final String veinNote;

  const Interpretation({
    required this.classificationNote,
    required this.healthScoreNote,
    required this.veinNote,
  });

  @override
  List<Object?> get props => [classificationNote, healthScoreNote, veinNote];
}
