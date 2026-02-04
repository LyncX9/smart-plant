import 'package:flutter/material.dart';

/// SmartPlant Vision Color Palette
/// Agricultural-tech theme with modern, clean aesthetics
class AppColors {
  AppColors._();

  // Primary Colors
  static const Color primaryGreen = Color(0xFF2ECC71);
  static const Color accentGreen = Color(0xFF00E676);
  static const Color darkGreen = Color(0xFF27AE60);
  
  // Gradients
  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [primaryGreen, accentGreen],
  );
  
  static const LinearGradient cardGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFE8F5E9), Color(0xFFC8E6C9)],
  );
  
  // Surface Colors
  static const Color surface = Color(0xFFFFFFFF);
  static const Color background = Color(0xFFF5F6FA);
  static const Color cardBackground = Color(0xFFFFFFFF);
  
  // Text Colors
  static const Color textPrimary = Color(0xFF212121);
  static const Color textSecondary = Color(0xFF757575);
  static const Color textLight = Color(0xFFFFFFFF);
  static const Color textMuted = Color(0xFFBDBDBD);
  
  // Status Colors
  static const Color success = Color(0xFF4CAF50);
  static const Color warning = Color(0xFFFFC107);
  static const Color error = Color(0xFFF44336);
  static const Color info = Color(0xFF2196F3);
  
  // Health Score Colors
  static const Color healthExcellent = Color(0xFF4CAF50);
  static const Color healthGood = Color(0xFF8BC34A);
  static const Color healthModerate = Color(0xFFFFC107);
  static const Color healthPoor = Color(0xFFFF9800);
  static const Color healthCritical = Color(0xFFF44336);
  
  /// Get color based on health score (0-100)
  static Color getHealthColor(double score) {
    if (score >= 80) return healthExcellent;
    if (score >= 60) return healthGood;
    if (score >= 40) return healthModerate;
    if (score >= 20) return healthPoor;
    return healthCritical;
  }
  
  /// Get color based on condition
  static Color getConditionColor(String condition) {
    switch (condition.toLowerCase()) {
      case 'healthy':
        return success;
      case 'diseased':
        return error;
      case 'uncertain':
        return warning;
      default:
        return textSecondary;
    }
  }
}
