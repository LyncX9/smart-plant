import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import '../../../domain/entities/scan.dart';
import '../../../domain/repositories/plant_repository.dart';

// Events
abstract class HistoryEvent extends Equatable {
  const HistoryEvent();
  
  @override
  List<Object?> get props => [];
}

class LoadHistory extends HistoryEvent {
  final bool refresh;
  const LoadHistory({this.refresh = false});
  
  @override
  List<Object?> get props => [refresh];
}

class LoadMoreHistory extends HistoryEvent {}

class ApplyFilter extends HistoryEvent {
  final String? plantType;
  final String? condition;
  final DateTime? dateFrom;
  final DateTime? dateTo;
  
  const ApplyFilter({
    this.plantType,
    this.condition,
    this.dateFrom,
    this.dateTo,
  });
  
  @override
  List<Object?> get props => [plantType, condition, dateFrom, dateTo];
}

class ClearFilters extends HistoryEvent {}

class LoadScanDetail extends HistoryEvent {
  final int scanId;
  const LoadScanDetail(this.scanId);
  
  @override
  List<Object?> get props => [scanId];
}

// States
abstract class HistoryState extends Equatable {
  const HistoryState();
  
  @override
  List<Object?> get props => [];
}

class HistoryInitial extends HistoryState {}

class HistoryLoading extends HistoryState {}

class HistoryLoaded extends HistoryState {
  final List<Scan> scans;
  final bool hasMore;
  final int currentPage;
  final HistoryFilter filter;
  
  const HistoryLoaded({
    required this.scans,
    this.hasMore = true,
    this.currentPage = 1,
    this.filter = const HistoryFilter(),
  });
  
  HistoryLoaded copyWith({
    List<Scan>? scans,
    bool? hasMore,
    int? currentPage,
    HistoryFilter? filter,
  }) {
    return HistoryLoaded(
      scans: scans ?? this.scans,
      hasMore: hasMore ?? this.hasMore,
      currentPage: currentPage ?? this.currentPage,
      filter: filter ?? this.filter,
    );
  }
  
  @override
  List<Object?> get props => [scans, hasMore, currentPage, filter];
}

class HistoryLoadingMore extends HistoryLoaded {
  const HistoryLoadingMore({
    required super.scans,
    super.hasMore,
    super.currentPage,
    super.filter,
  });
}

class HistoryError extends HistoryState {
  final String message;
  const HistoryError(this.message);
  
  @override
  List<Object?> get props => [message];
}

class ScanDetailLoading extends HistoryState {}

class ScanDetailLoaded extends HistoryState {
  final Scan scan;
  const ScanDetailLoaded(this.scan);
  
  @override
  List<Object?> get props => [scan];
}

class ScanDetailError extends HistoryState {
  final String message;
  const ScanDetailError(this.message);
  
  @override
  List<Object?> get props => [message];
}

// Filter model
class HistoryFilter extends Equatable {
  final String? plantType;
  final String? condition;
  final DateTime? dateFrom;
  final DateTime? dateTo;
  
  const HistoryFilter({
    this.plantType,
    this.condition,
    this.dateFrom,
    this.dateTo,
  });
  
  bool get hasFilters =>
      plantType != null || condition != null || dateFrom != null || dateTo != null;
  
  HistoryFilter copyWith({
    String? plantType,
    String? condition,
    DateTime? dateFrom,
    DateTime? dateTo,
  }) {
    return HistoryFilter(
      plantType: plantType ?? this.plantType,
      condition: condition ?? this.condition,
      dateFrom: dateFrom ?? this.dateFrom,
      dateTo: dateTo ?? this.dateTo,
    );
  }
  
  @override
  List<Object?> get props => [plantType, condition, dateFrom, dateTo];
}

// BLoC
class HistoryBloc extends Bloc<HistoryEvent, HistoryState> {
  final PlantRepository _repository;
  static const int _pageSize = 15;
  
  HistoryBloc({required PlantRepository repository})
      : _repository = repository,
        super(HistoryInitial()) {
    on<LoadHistory>(_onLoadHistory);
    on<LoadMoreHistory>(_onLoadMoreHistory);
    on<ApplyFilter>(_onApplyFilter);
    on<ClearFilters>(_onClearFilters);
    on<LoadScanDetail>(_onLoadScanDetail);
  }
  
  Future<void> _onLoadHistory(LoadHistory event, Emitter<HistoryState> emit) async {
    final currentState = state;
    final filter = currentState is HistoryLoaded && !event.refresh
        ? currentState.filter
        : const HistoryFilter();
    
    emit(HistoryLoading());
    
    try {
      final scans = await _repository.getHistory(
        plantType: filter.plantType,
        condition: filter.condition,
        dateFrom: filter.dateFrom,
        dateTo: filter.dateTo,
        page: 1,
        perPage: _pageSize,
      );
      
      emit(HistoryLoaded(
        scans: scans,
        hasMore: scans.length >= _pageSize,
        currentPage: 1,
        filter: filter,
      ));
    } catch (e) {
      // Handle 404 or network errors gracefully - show empty state instead of error
      // This happens when backend doesn't have /history endpoint (cloud deployment)
      if (e.toString().contains('404') || e.toString().contains('Server error')) {
        emit(const HistoryLoaded(
          scans: [],
          hasMore: false,
          currentPage: 1,
          filter: HistoryFilter(),
        ));
      } else {
        emit(HistoryError(e.toString()));
      }
    }
  }
  
  Future<void> _onLoadMoreHistory(LoadMoreHistory event, Emitter<HistoryState> emit) async {
    final currentState = state;
    if (currentState is! HistoryLoaded || !currentState.hasMore) return;
    if (currentState is HistoryLoadingMore) return;
    
    emit(HistoryLoadingMore(
      scans: currentState.scans,
      hasMore: currentState.hasMore,
      currentPage: currentState.currentPage,
      filter: currentState.filter,
    ));
    
    try {
      final nextPage = currentState.currentPage + 1;
      final newScans = await _repository.getHistory(
        plantType: currentState.filter.plantType,
        condition: currentState.filter.condition,
        dateFrom: currentState.filter.dateFrom,
        dateTo: currentState.filter.dateTo,
        page: nextPage,
        perPage: _pageSize,
      );
      
      emit(HistoryLoaded(
        scans: [...currentState.scans, ...newScans],
        hasMore: newScans.length >= _pageSize,
        currentPage: nextPage,
        filter: currentState.filter,
      ));
    } catch (e) {
      emit(HistoryError(e.toString()));
    }
  }
  
  Future<void> _onApplyFilter(ApplyFilter event, Emitter<HistoryState> emit) async {
    final filter = HistoryFilter(
      plantType: event.plantType,
      condition: event.condition,
      dateFrom: event.dateFrom,
      dateTo: event.dateTo,
    );
    
    emit(HistoryLoading());
    
    try {
      final scans = await _repository.getHistory(
        plantType: filter.plantType,
        condition: filter.condition,
        dateFrom: filter.dateFrom,
        dateTo: filter.dateTo,
        page: 1,
        perPage: _pageSize,
      );
      
      emit(HistoryLoaded(
        scans: scans,
        hasMore: scans.length >= _pageSize,
        currentPage: 1,
        filter: filter,
      ));
    } catch (e) {
      emit(HistoryError(e.toString()));
    }
  }
  
  void _onClearFilters(ClearFilters event, Emitter<HistoryState> emit) {
    add(const LoadHistory(refresh: true));
  }
  
  Future<void> _onLoadScanDetail(LoadScanDetail event, Emitter<HistoryState> emit) async {
    emit(ScanDetailLoading());
    
    try {
      final scan = await _repository.getScanById(event.scanId);
      emit(ScanDetailLoaded(scan));
    } catch (e) {
      emit(ScanDetailError(e.toString()));
    }
  }
}
