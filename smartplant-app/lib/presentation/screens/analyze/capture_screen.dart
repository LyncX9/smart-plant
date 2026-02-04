import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
import 'package:get_it/get_it.dart';

import '../../../core/constants/colors.dart';
import '../../../domain/repositories/plant_repository.dart';
import '../../bloc/analyze/analyze_bloc.dart';
import 'result_screen.dart';

class CaptureScreen extends StatelessWidget {
  const CaptureScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => AnalyzeBloc(
        repository: GetIt.instance<PlantRepository>(),
      ),
      child: const _CaptureScreenContent(),
    );
  }
}

class _CaptureScreenContent extends StatefulWidget {
  const _CaptureScreenContent();

  @override
  State<_CaptureScreenContent> createState() => _CaptureScreenContentState();
}

class _CaptureScreenContentState extends State<_CaptureScreenContent>
    with AutomaticKeepAliveClientMixin, WidgetsBindingObserver {
  final ImagePicker _picker = ImagePicker();

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle app resume after camera to prevent state loss
    if (state == AppLifecycleState.resumed && mounted) {
      setState(() {}); // Refresh UI
    }
  }

  /// Pick single image from camera
  Future<void> _pickImageFromCamera() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
        maxWidth: 1920,
        maxHeight: 1920,
      );
      
      if (image != null && mounted) {
        context.read<AnalyzeBloc>().add(AddImage(File(image.path)));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to capture image: $e')),
        );
      }
    }
  }

  /// Pick multiple images from gallery (up to remaining slots, max 7 total)
  Future<void> _pickMultipleImagesFromGallery() async {
    try {
      final currentCount = context.read<AnalyzeBloc>().state.imageCount;
      final remainingSlots = 7 - currentCount;
      
      if (remainingSlots <= 0) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Maximum 7 images allowed')),
        );
        return;
      }
      
      final List<XFile> images = await _picker.pickMultiImage(
        imageQuality: 85,
        maxWidth: 1920,
        maxHeight: 1920,
        limit: remainingSlots,
      );
      
      if (images.isNotEmpty && mounted) {
        for (final img in images) {
          context.read<AnalyzeBloc>().add(AddImage(File(img.path)));
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to pick images: $e')),
        );
      }
    }
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Add Leaf Image',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: _buildSourceOption(
                    icon: Icons.camera_alt,
                    label: 'Camera',
                    onTap: () {
                      Navigator.pop(context);
                      _pickImageFromCamera();
                    },
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _buildSourceOption(
                    icon: Icons.photo_library,
                    label: 'Gallery',
                    subtitle: 'Select multiple',
                    onTap: () {
                      Navigator.pop(context);
                      _pickMultipleImagesFromGallery();
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildSourceOption({
    required IconData icon,
    required String label,
    String? subtitle,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 24),
        decoration: BoxDecoration(
          color: AppColors.background,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            Icon(icon, size: 32, color: AppColors.primaryGreen),
            const SizedBox(height: 8),
            Text(label, style: Theme.of(context).textTheme.titleMedium),
            if (subtitle != null) ...[
              const SizedBox(height: 4),
              Text(
                subtitle,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Show dialog when rice leaf not detected (Unknown Object)
  void _showInvalidImageDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
            SizedBox(width: 12),
            Expanded(child: Text('Rice Leaf Not Detected')),
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
            onPressed: () {
              Navigator.pop(ctx); // Close dialog
              context.read<AnalyzeBloc>().add(ClearImages());
            },
            child: const Text('OK'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx); // Close dialog
              context.read<AnalyzeBloc>().add(ClearImages());
              Future.delayed(const Duration(milliseconds: 100), () {
                if (mounted) _showImageSourceDialog();
              });
            },
            child: const Text('Add Images'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // Required for AutomaticKeepAliveClientMixin
    
    return BlocConsumer<AnalyzeBloc, AnalyzeState>(
      listener: (context, state) {
        if (state is AnalyzeSuccess) {
          // Check for Unknown Object BEFORE navigating to ResultScreen
          final condition = state.result.summary.condition;
          if (condition == "Unknown Object" || condition == "Unknown") {
            _showInvalidImageDialog();
          } else {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (_) => ResultScreen(scan: state.result),
              ),
            );
          }
        } else if (state is AnalyzeError) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(state.message),
              backgroundColor: AppColors.error,
            ),
          );
        }
      },
      builder: (context, state) {
        final isLoading = state is AnalyzeLoading;
        
        return Scaffold(
          backgroundColor: AppColors.background,
          appBar: AppBar(
            title: const Text('Capture Leaves'),
            backgroundColor: Colors.white,
            elevation: 0,
            actions: [
              if (state.images.isNotEmpty)
                TextButton(
                  onPressed: () => context.read<AnalyzeBloc>().add(ClearImages()),
                  child: const Text('Clear All'),
                ),
            ],
          ),
          body: Stack(
            children: [
              Column(
                children: [
                  // Progress Indicator
                  _buildProgressSection(context, state),
                  
                  // Image Grid
                  Expanded(
                    child: state.images.isEmpty
                        ? _buildEmptyState(context)
                        : _buildImageGrid(context, state),
                  ),
                  
                  // Bottom Actions
                  _buildBottomActions(context, state),
                ],
              ),
              
              // Loading Overlay
              if (isLoading)
                Container(
                  color: Colors.black.withOpacity(0.5),
                  child: Center(
                    child: Container(
                      padding: const EdgeInsets.all(32),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CircularProgressIndicator(
                            valueColor: AlwaysStoppedAnimation(AppColors.primaryGreen),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            (state as AnalyzeLoading).message,
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildProgressSection(BuildContext context, AnalyzeState state) {
    return Container(
      padding: const EdgeInsets.all(16),
      color: Colors.white,
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '${state.imageCount} of 3-7 images',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: state.canStartAnalysis
                      ? AppColors.success.withOpacity(0.1)
                      : AppColors.warning.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  state.canStartAnalysis ? 'Ready' : 'Need ${3 - state.imageCount} more',
                  style: TextStyle(
                    color: state.canStartAnalysis ? AppColors.success : AppColors.warning,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: state.imageCount / 7,
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation(
                state.imageCount >= 3 ? AppColors.primaryGreen : AppColors.warning,
              ),
              minHeight: 6,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AppColors.primaryGreen.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.add_photo_alternate,
              size: 64,
              color: AppColors.primaryGreen,
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'Add Leaf Images',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Capture 3-7 photos of your plant leaves\nfor accurate health analysis',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            onPressed: _showImageSourceDialog,
            icon: const Icon(Icons.add_a_photo),
            label: const Text('Add First Image'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageGrid(BuildContext context, AnalyzeState state) {
    return GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1,
      ),
      itemCount: state.images.length,
      itemBuilder: (context, index) {
        final image = state.images[index];
        return Stack(
          children: [
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.file(
                  image,
                  fit: BoxFit.cover,
                  width: double.infinity,
                  height: double.infinity,
                ),
              ),
            ),
            Positioned(
              top: 8,
              left: 8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'Leaf ${index + 1}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
            Positioned(
              top: 8,
              right: 8,
              child: GestureDetector(
                onTap: () => context.read<AnalyzeBloc>().add(RemoveImage(index)),
                child: Container(
                  padding: const EdgeInsets.all(6),
                  decoration: const BoxDecoration(
                    color: Colors.red,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.close,
                    color: Colors.white,
                    size: 16,
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildBottomActions(BuildContext context, AnalyzeState state) {
    final isLoading = state is AnalyzeLoading;
    
    return Container(
      padding: const EdgeInsets.all(20),
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
            // Add Image Button
            if (state.canAddMore)
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: isLoading ? null : _showImageSourceDialog,
                  icon: const Icon(Icons.add_a_photo),
                  label: const Text('Add'),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    side: const BorderSide(color: AppColors.primaryGreen),
                  ),
                ),
              ),
            
            if (state.canAddMore)
              const SizedBox(width: 12),
            
            // Analyze Button
            Expanded(
              flex: 2,
              child: ElevatedButton.icon(
                onPressed: state.canStartAnalysis && !isLoading
                    ? () => context.read<AnalyzeBloc>().add(const StartAnalysis())
                    : null,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  disabledBackgroundColor: Colors.grey[300],
                ),
                icon: const Icon(Icons.auto_awesome),
                label: Text(
                  state.canStartAnalysis
                      ? 'Analyze (${state.imageCount})'
                      : 'Need ${3 - state.imageCount} More',
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
