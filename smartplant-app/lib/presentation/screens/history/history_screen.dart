import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:get_it/get_it.dart';
import 'package:intl/intl.dart';

import '../../../core/constants/colors.dart';
import '../../../domain/entities/scan.dart';
import '../../../domain/repositories/plant_repository.dart';
import '../../bloc/history/history_bloc.dart';
import 'history_detail_screen.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => HistoryBloc(
        repository: GetIt.instance<PlantRepository>(),
      )..add(const LoadHistory()),
      child: const _HistoryScreenContent(),
    );
  }
}

class _HistoryScreenContent extends StatelessWidget {
  const _HistoryScreenContent();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Scan History'),
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          BlocBuilder<HistoryBloc, HistoryState>(
            builder: (context, state) {
              if (state is HistoryLoaded && state.filter.hasFilters) {
                return TextButton.icon(
                  onPressed: () => context.read<HistoryBloc>().add(ClearFilters()),
                  icon: const Icon(Icons.clear, size: 18),
                  label: const Text('Clear'),
                );
              }
              return const SizedBox.shrink();
            },
          ),
        ],
      ),
      body: Column(
        children: [
          _buildFilterSection(context),
          Expanded(
            child: BlocBuilder<HistoryBloc, HistoryState>(
              builder: (context, state) {
                if (state is HistoryLoading) {
                  return const Center(
                    child: CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation(AppColors.primaryGreen),
                    ),
                  );
                }
                
                if (state is HistoryError) {
                  return _buildErrorState(context, state.message);
                }
                
                if (state is HistoryLoaded) {
                  if (state.scans.isEmpty) {
                    return _buildEmptyState(context);
                  }
                  return _buildScanList(context, state);
                }
                
                return const SizedBox.shrink();
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterSection(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      color: Colors.white,
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: BlocBuilder<HistoryBloc, HistoryState>(
          builder: (context, state) {
            final filter = state is HistoryLoaded ? state.filter : const HistoryFilter();
            
            return Row(
              children: [
                _buildFilterChip(
                  context,
                  icon: Icons.grass,
                  label: 'Rice',
                  isSelected: filter.plantType == 'rice',
                  onTap: () => context.read<HistoryBloc>().add(
                        ApplyFilter(
                          plantType: filter.plantType == 'rice' ? null : 'rice',
                          condition: filter.condition,
                          dateFrom: filter.dateFrom,
                          dateTo: filter.dateTo,
                        ),
                      ),
                ),
                const SizedBox(width: 8),
                _buildFilterChip(
                  context,
                  icon: Icons.check_circle,
                  label: 'Healthy',
                  isSelected: filter.condition == 'Healthy',
                  color: AppColors.success,
                  onTap: () => context.read<HistoryBloc>().add(
                        ApplyFilter(
                          plantType: filter.plantType,
                          condition: filter.condition == 'Healthy' ? null : 'Healthy',
                          dateFrom: filter.dateFrom,
                          dateTo: filter.dateTo,
                        ),
                      ),
                ),
                const SizedBox(width: 8),
                _buildFilterChip(
                  context,
                  icon: Icons.warning,
                  label: 'Diseased',
                  isSelected: filter.condition == 'Diseased',
                  color: AppColors.error,
                  onTap: () => context.read<HistoryBloc>().add(
                        ApplyFilter(
                          plantType: filter.plantType,
                          condition: filter.condition == 'Diseased' ? null : 'Diseased',
                          dateFrom: filter.dateFrom,
                          dateTo: filter.dateTo,
                        ),
                      ),
                ),
                const SizedBox(width: 8),
                _buildFilterChip(
                  context,
                  icon: Icons.help_outline,
                  label: 'Uncertain',
                  isSelected: filter.condition == 'Uncertain',
                  color: AppColors.warning,
                  onTap: () => context.read<HistoryBloc>().add(
                        ApplyFilter(
                          plantType: filter.plantType,
                          condition: filter.condition == 'Uncertain' ? null : 'Uncertain',
                          dateFrom: filter.dateFrom,
                          dateTo: filter.dateTo,
                        ),
                      ),
                ),
                const SizedBox(width: 8),
                _buildFilterChip(
                  context,
                  icon: Icons.calendar_today,
                  label: 'Date Range',
                  isSelected: filter.dateFrom != null || filter.dateTo != null,
                  onTap: () => _showDateRangePicker(context, filter),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildFilterChip(
    BuildContext context, {
    required IconData icon,
    required String label,
    required bool isSelected,
    Color? color,
    required VoidCallback onTap,
  }) {
    final chipColor = color ?? AppColors.primaryGreen;
    
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? chipColor.withOpacity(0.1) : AppColors.background,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? chipColor : Colors.grey.shade300,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: isSelected ? chipColor : AppColors.textSecondary,
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: isSelected ? chipColor : AppColors.textSecondary,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showDateRangePicker(BuildContext context, HistoryFilter filter) async {
    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
      initialDateRange: filter.dateFrom != null
          ? DateTimeRange(
              start: filter.dateFrom!,
              end: filter.dateTo ?? DateTime.now(),
            )
          : null,
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: AppColors.primaryGreen,
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null && context.mounted) {
      context.read<HistoryBloc>().add(
            ApplyFilter(
              plantType: filter.plantType,
              condition: filter.condition,
              dateFrom: picked.start,
              dateTo: picked.end,
            ),
          );
    }
  }

  Widget _buildScanList(BuildContext context, HistoryLoaded state) {
    return RefreshIndicator(
      onRefresh: () async {
        context.read<HistoryBloc>().add(const LoadHistory(refresh: true));
        await Future.delayed(const Duration(seconds: 1));
      },
      color: AppColors.primaryGreen,
      child: NotificationListener<ScrollNotification>(
        onNotification: (notification) {
          if (notification is ScrollEndNotification &&
              notification.metrics.pixels >= notification.metrics.maxScrollExtent - 200) {
            if (state.hasMore && state is! HistoryLoadingMore) {
              context.read<HistoryBloc>().add(LoadMoreHistory());
            }
          }
          return false;
        },
        child: ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: state.scans.length + (state.hasMore ? 1 : 0),
          itemBuilder: (context, index) {
            if (index >= state.scans.length) {
              return const Padding(
                padding: EdgeInsets.all(16),
                child: Center(
                  child: CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation(AppColors.primaryGreen),
                  ),
                ),
              );
            }
            return _buildScanCard(context, state.scans[index]);
          },
        ),
      ),
    );
  }

  Widget _buildScanCard(BuildContext context, Scan scan) {
    final summary = scan.summary;
    final scoreColor = AppColors.getHealthColor(summary.healthScore);
    
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => HistoryDetailScreen(scanId: scan.id),
        ),
      ),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
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
        child: Row(
          children: [
            // Health Score Mini Gauge
            Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: scoreColor, width: 3),
              ),
              child: Center(
                child: Text(
                  '${summary.healthScore.toStringAsFixed(0)}',
                  style: TextStyle(
                    color: scoreColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            
            // Info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppColors.getConditionColor(summary.condition).withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          summary.condition,
                          style: TextStyle(
                            color: AppColors.getConditionColor(summary.condition),
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const Spacer(),
                      Text(
                        DateFormat('MMM d, y').format(scan.timestamp),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: AppColors.textSecondary,
                            ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    summary.predictedClass,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Icon(Icons.grass, size: 14, color: AppColors.textSecondary),
                      const SizedBox(width: 4),
                      Text(
                        scan.plantType.toUpperCase(),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: AppColors.textSecondary,
                            ),
                      ),
                      const SizedBox(width: 12),
                      Icon(Icons.eco, size: 14, color: AppColors.textSecondary),
                      const SizedBox(width: 4),
                      Text(
                        '${scan.leaves.isNotEmpty ? scan.leaves.length : '?'} leaves',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: AppColors.textSecondary,
                            ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            
            const Icon(Icons.chevron_right, color: AppColors.textSecondary),
          ],
        ),
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
              Icons.history,
              size: 64,
              color: AppColors.primaryGreen,
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'No Scans Yet',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Your plant analysis history will appear here',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState(BuildContext context, String message) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 64, color: AppColors.error),
          const SizedBox(height: 16),
          Text(
            'Oops! Something went wrong',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              message,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppColors.textSecondary,
                  ),
            ),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () => context.read<HistoryBloc>().add(const LoadHistory()),
            icon: const Icon(Icons.refresh),
            label: const Text('Try Again'),
          ),
        ],
      ),
    );
  }
}
