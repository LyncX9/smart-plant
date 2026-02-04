import 'package:equatable/equatable.dart';

/// Leaf analysis result entity
class LeafResult extends Equatable {
  final int leafIndex;
  final String filename;
  final String predictedClass;
  final double confidence;
  final int lesionCount;
  final double lesionAreaPercent;
  final String veinStatus;
  final VeinMetrics veinMetrics;
  final String? originalUrl;
  final String? overlayUrl;

  const LeafResult({
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

  @override
  List<Object?> get props => [
        leafIndex,
        filename,
        predictedClass,
        confidence,
        lesionCount,
        lesionAreaPercent,
        veinStatus,
        veinMetrics,
        originalUrl,
        overlayUrl,
      ];
}

/// Vein morphometry metrics (for display only - not diagnostic)
class VeinMetrics extends Equatable {
  final int? lengthPx;
  final double? densityPercent;
  final double? continuity;

  const VeinMetrics({
    this.lengthPx,
    this.densityPercent,
    this.continuity,
  });

  @override
  List<Object?> get props => [lengthPx, densityPercent, continuity];
}
