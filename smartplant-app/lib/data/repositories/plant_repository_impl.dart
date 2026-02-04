import 'dart:io';
import '../../domain/entities/scan.dart';
import '../../domain/repositories/plant_repository.dart';
import '../datasources/plant_api_datasource.dart';

/// Implementation of PlantRepository using remote API
class PlantRepositoryImpl implements PlantRepository {
  final PlantApiDataSource _dataSource;

  PlantRepositoryImpl({PlantApiDataSource? dataSource})
      : _dataSource = dataSource ?? PlantApiDataSource();

  @override
  Future<Scan> analyzePlant({
    required List<File> images,
    String plantType = 'rice',
  }) async {
    final model = await _dataSource.analyzePlant(
      images: images,
      plantType: plantType,
    );
    return model.toEntity();
  }

  @override
  Future<List<Scan>> getHistory({
    String? plantType,
    String? condition,
    DateTime? dateFrom,
    DateTime? dateTo,
    int page = 1,
    int perPage = 15,
  }) async {
    final models = await _dataSource.getHistory(
      plantType: plantType,
      condition: condition,
      dateFrom: dateFrom,
      dateTo: dateTo,
      page: page,
      perPage: perPage,
    );
    return models.map((m) => m.toEntity()).toList();
  }

  @override
  Future<Scan> getScanById(int id) async {
    final model = await _dataSource.getScanById(id);
    return model.toEntity();
  }
}
