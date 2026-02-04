import '../../domain/entities/leaf_result.dart';

/// Model for parsing leaf result from API response
class LeafResultModel {
  final int leafIndex;
  final String filename;
  final String predictedClass;
  final double confidence;
  final int lesionCount;
  final double lesionAreaPercent;
  final String veinStatus;
  final VeinMetricsModel veinMetrics;
  final String? originalUrl;
  final String? overlayUrl;

  LeafResultModel({
    required this.leafIndex,
    required this.filename,
    required this.predictedClass,
    required this.confidence,
    required this.lesionCount,
    required this.lesionAreaPercent,
    required this.veinStatus,
    required this.veinMetrics,
    this.originalUrl,
    this.overlayUrl,
  });

  factory LeafResultModel.fromJson(Map<String, dynamic> json) {
    return LeafResultModel(
      leafIndex: json['leaf_index'] ?? 0,
      filename: json['filename'] ?? '',
      predictedClass: json['predicted_class'] ?? 'Unknown',
      confidence: (json['confidence'] ?? 0).toDouble(),
      lesionCount: json['lesion_count'] ?? 0,
      lesionAreaPercent: (json['lesion_area_percent'] ?? 0).toDouble(),
      veinStatus: json['vein_status'] ?? 'Unknown',
      veinMetrics: VeinMetricsModel.fromJson(json['vein_metrics'] ?? {}),
      originalUrl: json['original_url'],
      overlayUrl: json['overlay_url'],
    );
  }

  LeafResult toEntity() {
    return LeafResult(
      leafIndex: leafIndex,
      filename: filename,
      predictedClass: predictedClass,
      confidence: confidence,
      lesionCount: lesionCount,
      lesionAreaPercent: lesionAreaPercent,
      veinStatus: veinStatus,
      veinMetrics: veinMetrics.toEntity(),
      originalUrl: originalUrl,
      overlayUrl: overlayUrl,
    );
  }
}

class VeinMetricsModel {
  final int? lengthPx;
  final double? densityPercent;
  final double? continuity;

  VeinMetricsModel({
    this.lengthPx,
    this.densityPercent,
    this.continuity,
  });

  factory VeinMetricsModel.fromJson(Map<String, dynamic> json) {
    return VeinMetricsModel(
      lengthPx: json['length_px'],
      densityPercent: json['density_percent']?.toDouble(),
      continuity: json['continuity']?.toDouble(),
    );
  }

  VeinMetrics toEntity() {
    return VeinMetrics(
      lengthPx: lengthPx,
      densityPercent: densityPercent,
      continuity: continuity,
    );
  }
}
