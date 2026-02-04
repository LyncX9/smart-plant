import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import 'package:cached_network_image/cached_network_image.dart';

import '../../../core/constants/colors.dart';
import '../../../domain/entities/scan.dart';
import '../../../domain/entities/leaf_result.dart';
import '../main_screen.dart';

class ResultScreen extends StatefulWidget {
  final Scan scan;

  const ResultScreen({super.key, required this.scan});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (widget.scan.summary.condition == "Unknown Object") {
        _showUnknownObjectDialog();
      }
    });
  }

  void _showUnknownObjectDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
            SizedBox(width: 12),
            Expanded(child: Text('Object Not Detected')),
          ],
        ),
        content: const Text(
          'Gambar tidak terdeteksi sebagai daun padi (Rice Leaf).\n\n'
          'Sistem saat ini hanya dioptimalkan untuk menganalisis daun padi. '
          'Mohon pastikan Anda mengambil foto daun padi yang jelas dan fokus.',
          style: TextStyle(height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pushAndRemoveUntil(
              context,
              MaterialPageRoute(builder: (_) => const MainScreen()),
              (route) => false,
            ),
            child: const Text('Back to Home'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Analysis Results'),
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (_) => const MainScreen()),
            (route) => false,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Share feature coming soon!')),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Health Score Section
            _buildHealthScoreSection(context),
            
            // Summary Cards
            _buildSummaryCards(context),
            
            // Vein Morphometry Info
            _buildVeinInfoSection(context),
            
            // Leaf Breakdown
            _buildLeafBreakdown(context),
            
            // Interpretation Notes
            _buildInterpretationSection(context),
            
            const SizedBox(height: 32),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomBar(context),
    );
  }

  Widget _buildHealthScoreSection(BuildContext context) {
    final summary = widget.scan.summary;
    final scoreColor = AppColors.getHealthColor(summary.healthScore);
    
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 32),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          // Circular Health Score
          CircularPercentIndicator(
            radius: 90,
            lineWidth: 12,
            percent: summary.healthScore / 100,
            center: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${summary.healthScore.toStringAsFixed(1)}',
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
            animation: true,
            animationDuration: 1000,
          ),
          const SizedBox(height: 24),
          
          // Condition Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            decoration: BoxDecoration(
              color: AppColors.getConditionColor(summary.condition).withOpacity(0.1),
              borderRadius: BorderRadius.circular(30),
              border: Border.all(
                color: AppColors.getConditionColor(summary.condition).withOpacity(0.5),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _getConditionIcon(summary.condition),
                  color: AppColors.getConditionColor(summary.condition),
                  size: 20,
                ),
                const SizedBox(width: 8),
                Text(
                  summary.condition,
                  style: TextStyle(
                    color: AppColors.getConditionColor(summary.condition),
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          
          // Plant Type & Prediction
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildInfoPill(
                icon: Icons.grass,
                label: widget.scan.plantType.toUpperCase(),
                color: AppColors.primaryGreen,
              ),
              const SizedBox(width: 12),
              _buildInfoPill(
                icon: Icons.psychology,
                label: summary.predictedClass,
                color: summary.isHealthy ? AppColors.success : AppColors.error,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInfoPill({
    required IconData icon,
    required String label,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  IconData _getConditionIcon(String condition) {
    switch (condition.toLowerCase()) {
      case 'healthy':
        return Icons.check_circle;
      case 'diseased':
        return Icons.warning;
      default:
        return Icons.help_outline;
    }
  }

  Widget _buildSummaryCards(BuildContext context) {
    final summary = widget.scan.summary;
    
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: _buildStatCard(
              context,
              icon: Icons.auto_awesome,
              label: 'AI Confidence',
              value: '${summary.confidencePercent.toStringAsFixed(1)}%',
              color: Colors.blue,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _buildStatCard(
              context,
              icon: Icons.eco,
              label: 'Leaves Analyzed',
              value: '${widget.scan.leaves.length}',
              color: AppColors.primaryGreen,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _buildStatCard(
              context,
              icon: Icons.bug_report,
              label: 'Lesions Found',
              value: '${summary.totalLesionCount}',
              color: summary.totalLesionCount > 0 ? Colors.orange : AppColors.success,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatCard(
    BuildContext context, {
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(height: 8),
          Text(
            value,
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildVeinInfoSection(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.blue.shade100),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.blue.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.info_outline, color: Colors.blue),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'About Vein Analysis',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        color: Colors.blue.shade700,
                      ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Vein morphometry is shown for visual support and education only. It does NOT affect the health score or disease diagnosis.',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.blue.shade600,
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLeafBreakdown(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Leaf Analysis Breakdown',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          ...widget.scan.leaves.map((leaf) => _buildLeafCard(context, leaf)),
        ],
      ),
    );
  }

  Widget _buildLeafCard(BuildContext context, LeafResult leaf) {
    final conditionColor = leaf.predictedClass.toLowerCase() == 'healthy'
        ? AppColors.success
        : AppColors.error;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          // Header
          ListTile(
            leading: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.primaryGreen.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Center(
                child: Text(
                  '${leaf.leafIndex}',
                  style: const TextStyle(
                    color: AppColors.primaryGreen,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
            title: Text('Leaf ${leaf.leafIndex}'),
            subtitle: Text(leaf.filename),
            trailing: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: conditionColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                leaf.predictedClass,
                style: TextStyle(
                  color: conditionColor,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
            ),
          ),
          
          // Details
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Row(
              children: [
                _buildLeafStat(
                  'Confidence',
                  '${(leaf.confidence * 100).toStringAsFixed(1)}%',
                  Icons.auto_awesome,
                ),
                const Spacer(),
                _buildLeafStat(
                  'Lesions',
                  '${leaf.lesionCount}',
                  Icons.bug_report,
                ),
                const Spacer(),
                _buildLeafStat(
                  'Vein Status',
                  leaf.veinStatus,
                  Icons.grain,
                ),
              ],
            ),
          ),
          
          // Images Row
          if (leaf.originalUrl != null || leaf.overlayUrl != null)
            Container(
              height: 100,
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Row(
                children: [
                  if (leaf.originalUrl != null)
                    Expanded(
                      child: _buildImagePreview(
                        context,
                        leaf.originalUrl!,
                        'Original',
                      ),
                    ),
                  if (leaf.originalUrl != null && leaf.overlayUrl != null)
                    const SizedBox(width: 12),
                  if (leaf.overlayUrl != null)
                    Expanded(
                      child: _buildImagePreview(
                        context,
                        leaf.overlayUrl!,
                        'Vein Overlay',
                      ),
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildLeafStat(String label, String value, IconData icon) {
    return Column(
      children: [
        Icon(icon, size: 16, color: AppColors.textSecondary),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        Text(
          label,
          style: TextStyle(fontSize: 10, color: AppColors.textSecondary),
        ),
      ],
    );
  }

  Widget _buildImagePreview(BuildContext context, String url, String label) {
    return GestureDetector(
      onTap: () {
        // Could implement full-screen image view here
        showDialog(
          context: context,
          builder: (ctx) => Dialog(
            backgroundColor: Colors.transparent,
            insetPadding: const EdgeInsets.all(16),
            child: InteractiveViewer(
              child: url.startsWith('data:image')
                  ? Image.memory(
                      base64Decode(url.split(',')[1]),
                      fit: BoxFit.contain,
                    )
                  : CachedNetworkImage(
                      imageUrl: url,
                      fit: BoxFit.contain,
                    ),
            ),
          ),
        );
      },
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: url.startsWith('data:image')
                  ? Image.memory(
                      base64Decode(url.split(',')[1]),
                      fit: BoxFit.cover,
                      width: double.infinity,
                      errorBuilder: (_, __, ___) => _buildErrorImage(),
                    )
                  : CachedNetworkImage(
                      imageUrl: url,
                      fit: BoxFit.cover,
                      width: double.infinity,
                      placeholder: (_, __) => _buildLoadingImage(),
                      errorWidget: (_, __, ___) => _buildErrorImage(),
                    ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingImage() {
    return Container(
      color: AppColors.background,
      child: const Center(
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
    );
  }

  Widget _buildErrorImage() {
    return Container(
      color: AppColors.background,
      child: const Icon(Icons.image_not_supported, color: AppColors.textSecondary),
    );
  }

  Widget _buildInterpretationSection(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.info_outline, size: 20, color: AppColors.textSecondary),
              const SizedBox(width: 8),
              Text(
                'Interpretation Notes',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      color: AppColors.textSecondary,
                    ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _buildNote('Classification:', widget.scan.interpretation.classificationNote),
          const SizedBox(height: 8),
          _buildNote('Health Score:', widget.scan.interpretation.healthScoreNote),
          const SizedBox(height: 8),
          _buildNote('Vein Analysis:', widget.scan.interpretation.veinNote),
        ],
      ),
    );
  }

  Widget _buildNote(String title, String text) {
    return RichText(
      text: TextSpan(
        style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
        children: [
          TextSpan(
            text: '$title ',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          TextSpan(text: text),
        ],
      ),
    );
  }

  Widget _buildBottomBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () => Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(builder: (_) => const MainScreen()),
                  (route) => false,
                ),
                icon: const Icon(Icons.home),
                label: const Text('Home'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              flex: 2,
              child: ElevatedButton.icon(
                onPressed: () {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (_) => const HomeScreen()),
                  );
                  Future.delayed(const Duration(milliseconds: 100), () {
                    if (context.mounted) {
                      // Navigate to capture screen for new scan
                    }
                  });
                },
                icon: const Icon(Icons.camera_alt),
                label: const Text('New Scan'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
