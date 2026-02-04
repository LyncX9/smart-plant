import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:get_it/get_it.dart';

import '../../../core/constants/colors.dart';
import '../../../domain/repositories/plant_repository.dart';
import '../../bloc/history/history_bloc.dart';
import '../analyze/result_screen.dart';

class HistoryDetailScreen extends StatelessWidget {
  final int scanId;

  const HistoryDetailScreen({super.key, required this.scanId});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => HistoryBloc(
        repository: GetIt.instance<PlantRepository>(),
      )..add(LoadScanDetail(scanId)),
      child: const _HistoryDetailContent(),
    );
  }
}

class _HistoryDetailContent extends StatelessWidget {
  const _HistoryDetailContent();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Scan Details'),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: BlocBuilder<HistoryBloc, HistoryState>(
        builder: (context, state) {
          if (state is ScanDetailLoading) {
            return const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation(AppColors.primaryGreen),
              ),
            );
          }
          
          if (state is ScanDetailError) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 64, color: AppColors.error),
                  const SizedBox(height: 16),
                  Text(
                    'Failed to load scan',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    state.message,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: AppColors.textSecondary,
                        ),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Go Back'),
                  ),
                ],
              ),
            );
          }
          
          if (state is ScanDetailLoaded) {
            // Reuse the ResultScreen for consistent display
            return ResultScreen(scan: state.scan);
          }
          
          return const SizedBox.shrink();
        },
      ),
    );
  }
}
