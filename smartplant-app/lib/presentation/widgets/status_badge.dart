import 'package:flutter/material.dart';
import '../../../core/constants/colors.dart';

/// Status badge widget for displaying condition status
class StatusBadge extends StatelessWidget {
  final String condition;
  final bool showIcon;
  final double fontSize;

  const StatusBadge({
    super.key,
    required this.condition,
    this.showIcon = true,
    this.fontSize = 12,
  });

  @override
  Widget build(BuildContext context) {
    final color = AppColors.getConditionColor(condition);
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showIcon) ...[
            Icon(_getIcon(), size: fontSize + 4, color: color),
            const SizedBox(width: 6),
          ],
          Text(
            condition,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w600,
              fontSize: fontSize,
            ),
          ),
        ],
      ),
    );
  }

  IconData _getIcon() {
    switch (condition.toLowerCase()) {
      case 'healthy':
        return Icons.check_circle;
      case 'diseased':
        return Icons.warning;
      case 'uncertain':
        return Icons.help_outline;
      default:
        return Icons.info_outline;
    }
  }
}
