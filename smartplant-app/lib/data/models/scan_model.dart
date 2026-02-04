import '../../domain/entities/scan.dart';
import 'leaf_result_model.dart';

/// Model for parsing scan response from API
class ScanModel {
  final int id;
  final DateTime timestamp;
  final String plantType;
  final String status;
  final ScanSummaryModel summary;
  final List<LeafResultModel> leaves;
  final InterpretationModel interpretation;

  ScanModel({
    required this.id,
    required this.timestamp,
    required this.plantType,
    required this.status,
    required this.summary,
    required this.leaves,
    required this.interpretation,
  });

  factory ScanModel.fromJson(Map<String, dynamic> json) {
    return ScanModel(
      id: json['scan_id'] ?? json['id'] ?? 0,
      timestamp: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
      plantType: json['plant_type'] ?? 'rice',
      status: json['status'] ?? 'completed',
      summary: ScanSummaryModel.fromJson(json['summary'] ?? {}),
      leaves: (json['leaves'] as List<dynamic>?)
              ?.map((e) => LeafResultModel.fromJson(e))
              .toList() ??
          [],
      interpretation: InterpretationModel.fromJson(json['interpretation'] ?? {}),
    );
  }

  /// Parse from list response (history list item)
  factory ScanModel.fromListJson(Map<String, dynamic> json) {
    return ScanModel(
      id: json['id'] ?? 0,
      timestamp: DateTime.parse(json['created_at'] ?? DateTime.now().toIso8601String()),
      plantType: json['plant_type'] ?? 'rice',
      status: 'completed',
      summary: ScanSummaryModel.fromListJson(json),
      leaves: [],
      interpretation: InterpretationModel.defaults(),
    );
  }

  Scan toEntity() {
    return Scan(
      id: id,
      timestamp: timestamp,
      plantType: plantType,
      status: status,
      summary: summary.toEntity(),
      leaves: leaves.map((e) => e.toEntity()).toList(),
      interpretation: interpretation.toEntity(),
    );
  }
}

class ScanSummaryModel {
  final double healthScore;
  final String condition;
  final String predictedClass;
  final double confidence;
  final Map<String, double> classProbabilities;
  final double avgLesionAreaPercent;
  final int totalLesionCount;

  ScanSummaryModel({
    required this.healthScore,
    required this.condition,
    required this.predictedClass,
    required this.confidence,
    required this.classProbabilities,
    required this.avgLesionAreaPercent,
    required this.totalLesionCount,
  });

  factory ScanSummaryModel.fromJson(Map<String, dynamic> json) {
    final probabilities = <String, double>{};
    if (json['class_probabilities'] != null) {
      (json['class_probabilities'] as Map<String, dynamic>).forEach((key, value) {
        probabilities[key] = (value ?? 0).toDouble();
      });
    }

    return ScanSummaryModel(
      healthScore: (json['health_score'] ?? 0).toDouble(),
      condition: json['condition'] ?? 'Unknown',
      predictedClass: json['predicted_class'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0).toDouble(),
      classProbabilities: probabilities,
      avgLesionAreaPercent: (json['avg_lesion_area_percent'] ?? 0).toDouble(),
      totalLesionCount: json['total_lesion_count'] ?? 0,
    );
  }

  /// Parse from list item (minimal data)
  factory ScanSummaryModel.fromListJson(Map<String, dynamic> json) {
    return ScanSummaryModel(
      healthScore: (json['health_score'] ?? 0).toDouble(),
      condition: json['condition'] ?? 'Unknown',
      predictedClass: json['predicted_class'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0).toDouble(),
      classProbabilities: {},
      avgLesionAreaPercent: 0,
      totalLesionCount: 0,
    );
  }

  ScanSummary toEntity() {
    return ScanSummary(
      healthScore: healthScore,
      condition: condition,
      predictedClass: predictedClass,
      confidence: confidence,
      classProbabilities: classProbabilities,
      avgLesionAreaPercent: avgLesionAreaPercent,
      totalLesionCount: totalLesionCount,
    );
  }
}

class InterpretationModel {
  final String classificationNote;
  final String healthScoreNote;
  final String veinNote;

  InterpretationModel({
    required this.classificationNote,
    required this.healthScoreNote,
    required this.veinNote,
  });

  factory InterpretationModel.fromJson(Map<String, dynamic> json) {
    return InterpretationModel(
      classificationNote: json['classification_note'] ?? 
          'Based solely on CNN deep learning analysis',
      healthScoreNote: json['health_score_note'] ?? 
          'Heuristic indicator (0-100), not a biological diagnosis',
      veinNote: json['vein_note'] ?? 
          'Vein analysis is for visual support only, not diagnostic',
    );
  }

  factory InterpretationModel.defaults() {
    return InterpretationModel(
      classificationNote: 'Based solely on CNN deep learning analysis',
      healthScoreNote: 'Heuristic indicator (0-100), not a biological diagnosis',
      veinNote: 'Vein analysis is for visual support only, not diagnostic',
    );
  }

  Interpretation toEntity() {
    return Interpretation(
      classificationNote: classificationNote,
      healthScoreNote: healthScoreNote,
      veinNote: veinNote,
    );
  }
}
