import 'dart:io';
import 'package:dio/dio.dart';
import '../../core/constants/api_endpoints.dart';
import '../models/scan_model.dart';

/// Remote data source for plant analysis API calls
class PlantApiDataSource {
  final Dio _dio;

  PlantApiDataSource({Dio? dio}) : _dio = dio ?? _createDio();

  static Dio _createDio() {
    final dio = Dio(BaseOptions(
      baseUrl: ApiEndpoints.baseUrl,
      connectTimeout: ApiEndpoints.connectTimeout,
      receiveTimeout: ApiEndpoints.receiveTimeout,
      sendTimeout: ApiEndpoints.sendTimeout,
      headers: {
        'Accept': 'application/json',
      },
    ));

    // Add logging in debug mode
    dio.interceptors.add(LogInterceptor(
      request: true,
      requestHeader: true,
      requestBody: true,
      responseHeader: false,
      responseBody: true,
      error: true,
    ));

    return dio;
  }

  /// Upload images for analysis
  Future<ScanModel> analyzePlant({
    required List<File> images,
    String plantType = 'rice',
  }) async {
    final formData = FormData();

    // Add plant type
    formData.fields.add(MapEntry('plant_type', plantType));

    // Add images
    for (int i = 0; i < images.length; i++) {
      final file = images[i];
      final fileName = '${plantType}_leaf_${i + 1}.jpg';
      formData.files.add(MapEntry(
        'images[]',
        await MultipartFile.fromFile(file.path, filename: fileName),
      ));
    }

    try {
      final response = await _dio.post(
        ApiEndpoints.analyze,
        data: formData,
        options: Options(
          contentType: 'multipart/form-data',
        ),
      );

      if (response.data['status'] == 'error') {
        throw ApiException(
          message: response.data['message'] ?? 'Analysis failed',
          statusCode: response.statusCode ?? 500,
        );
      }

      return ScanModel.fromJson(response.data);
    } on DioException catch (e) {
      throw _handleDioError(e);
    }
  }

  /// Get scan history with filters
  Future<List<ScanModel>> getHistory({
    String? plantType,
    String? condition,
    DateTime? dateFrom,
    DateTime? dateTo,
    int page = 1,
    int perPage = 15,
  }) async {
    final queryParams = <String, dynamic>{
      'page': page,
      'per_page': perPage,
    };

    if (plantType != null) queryParams['plant_type'] = plantType;
    if (condition != null) queryParams['condition'] = condition;
    if (dateFrom != null) queryParams['date_from'] = dateFrom.toIso8601String().split('T')[0];
    if (dateTo != null) queryParams['date_to'] = dateTo.toIso8601String().split('T')[0];

    try {
      final response = await _dio.get(
        ApiEndpoints.history,
        queryParameters: queryParams,
      );

      final data = response.data['data'] as List<dynamic>? ?? [];
      return data.map((e) => ScanModel.fromListJson(e)).toList();
    } on DioException catch (e) {
      throw _handleDioError(e);
    }
  }

  /// Get single scan by ID
  Future<ScanModel> getScanById(int id) async {
    try {
      final response = await _dio.get(ApiEndpoints.historyDetail(id));
      return ScanModel.fromJson(response.data['data']);
    } on DioException catch (e) {
      throw _handleDioError(e);
    }
  }

  /// Handle Dio errors and convert to ApiException
  ApiException _handleDioError(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        return ApiException(
          message: 'Connection timeout. Please try again.',
          statusCode: 408,
        );
      case DioExceptionType.connectionError:
        return ApiException(
          message: 'No internet connection. Please check your network.',
          statusCode: 0,
        );
      case DioExceptionType.badResponse:
        final statusCode = e.response?.statusCode ?? 500;
        final message = e.response?.data?['message'] ?? 'Server error occurred';
        return ApiException(message: message, statusCode: statusCode);
      default:
        return ApiException(
          message: 'An unexpected error occurred',
          statusCode: 500,
        );
    }
  }
}

/// Custom API exception
class ApiException implements Exception {
  final String message;
  final int statusCode;

  ApiException({required this.message, required this.statusCode});

  @override
  String toString() => 'ApiException: [$statusCode] $message';
}
