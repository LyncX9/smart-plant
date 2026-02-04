/// API Configuration
class ApiEndpoints {
  ApiEndpoints._();
  
  /// Base URL for the SmartPlant API
  /// Change this to your production URL
  // static const String baseUrl = 'http://10.0.2.2:8000/api';  // Android emulator
  // static const String baseUrl = 'http://192.168.1.3:8000/api';  // Local Wi-Fi
  static const String baseUrl = 'https://smart-plant.onrender.com';  // Production (Render)
  
  /// Endpoints
  static const String analyze = '/analyze';
  static const String history = '/history';
  static String historyDetail(int id) => '/history/$id';
  static const String health = '/health';
  
  /// Timeouts
  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 120);
  static const Duration sendTimeout = Duration(seconds: 120);
}
