import 'dart:io';
import '../entities/scan.dart';

/// Repository interface for plant analysis operations
abstract class PlantRepository {
  /// Analyze plant images and return the scan result
  Future<Scan> analyzePlant({
    required List<File> images,
    String plantType = 'rice',
  });

  /// Get list of scan history with optional filters
  Future<List<Scan>> getHistory({
    String? plantType,
    String? condition,
    DateTime? dateFrom,
    DateTime? dateTo,
    int page = 1,
    int perPage = 15,
  });

  /// Get a single scan by ID
  Future<Scan> getScanById(int id);
}
