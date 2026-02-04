import 'package:flutter/material.dart';
import '../../../core/constants/colors.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

/// Animated circular health score gauge widget
class HealthScoreGauge extends StatelessWidget {
  final double score;
  final double radius;
  final double lineWidth;
  final bool animate;

  const HealthScoreGauge({
    super.key,
    required this.score,
    this.radius = 90,
    this.lineWidth = 12,
    this.animate = true,
  });

  @override
  Widget build(BuildContext context) {
    final scoreColor = AppColors.getHealthColor(score);
    
    return CircularPercentIndicator(
      radius: radius,
      lineWidth: lineWidth,
      percent: score / 100,
      center: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            score.toStringAsFixed(1),
            style: Theme.of(context).textTheme.displayMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: scoreColor,
                ),
          ),
          Text(
            'Health Score',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
        ],
      ),
      progressColor: scoreColor,
      backgroundColor: scoreColor.withOpacity(0.2),
      circularStrokeCap: CircularStrokeCap.round,
      animation: animate,
      animationDuration: 1000,
    );
  }
}
