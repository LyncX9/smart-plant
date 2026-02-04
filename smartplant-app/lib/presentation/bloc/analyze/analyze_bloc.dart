import 'dart:io';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import '../../../domain/entities/scan.dart';
import '../../../domain/repositories/plant_repository.dart';

// Events
abstract class AnalyzeEvent extends Equatable {
  const AnalyzeEvent();
  
  @override
  List<Object?> get props => [];
}

class AddImage extends AnalyzeEvent {
  final File image;
  const AddImage(this.image);
  
  @override
  List<Object?> get props => [image];
}

class RemoveImage extends AnalyzeEvent {
  final int index;
  const RemoveImage(this.index);
  
  @override
  List<Object?> get props => [index];
}

class ClearImages extends AnalyzeEvent {}

class StartAnalysis extends AnalyzeEvent {
  final String plantType;
  const StartAnalysis({this.plantType = 'rice'});
  
  @override
  List<Object?> get props => [plantType];
}

class ResetAnalysis extends AnalyzeEvent {}

// States
abstract class AnalyzeState extends Equatable {
  final List<File> images;
  
  const AnalyzeState({this.images = const []});
  
  @override
  List<Object?> get props => [images];
  
  bool get canAddMore => images.length < 7;
  bool get canStartAnalysis => images.length >= 3;
  int get imageCount => images.length;
}

class AnalyzeInitial extends AnalyzeState {
  const AnalyzeInitial({super.images});
}

class AnalyzeImagesUpdated extends AnalyzeState {
  const AnalyzeImagesUpdated({required List<File> images}) : super(images: images);
}

class AnalyzeLoading extends AnalyzeState {
  final String message;
  
  const AnalyzeLoading({
    required List<File> images,
    this.message = 'Analyzing your plant...',
  }) : super(images: images);
  
  @override
  List<Object?> get props => [...super.props, message];
}

class AnalyzeSuccess extends AnalyzeState {
  final Scan result;
  
  const AnalyzeSuccess({
    required List<File> images,
    required this.result,
  }) : super(images: images);
  
  @override
  List<Object?> get props => [...super.props, result];
}

class AnalyzeError extends AnalyzeState {
  final String message;
  
  const AnalyzeError({
    required List<File> images,
    required this.message,
  }) : super(images: images);
  
  @override
  List<Object?> get props => [...super.props, message];
}

// BLoC
class AnalyzeBloc extends Bloc<AnalyzeEvent, AnalyzeState> {
  final PlantRepository _repository;
  
  AnalyzeBloc({required PlantRepository repository})
      : _repository = repository,
        super(const AnalyzeInitial()) {
    on<AddImage>(_onAddImage);
    on<RemoveImage>(_onRemoveImage);
    on<ClearImages>(_onClearImages);
    on<StartAnalysis>(_onStartAnalysis);
    on<ResetAnalysis>(_onResetAnalysis);
  }
  
  void _onAddImage(AddImage event, Emitter<AnalyzeState> emit) {
    if (!state.canAddMore) return;
    
    final newImages = [...state.images, event.image];
    emit(AnalyzeImagesUpdated(images: newImages));
  }
  
  void _onRemoveImage(RemoveImage event, Emitter<AnalyzeState> emit) {
    if (event.index < 0 || event.index >= state.images.length) return;
    
    final newImages = [...state.images]..removeAt(event.index);
    emit(AnalyzeImagesUpdated(images: newImages));
  }
  
  void _onClearImages(ClearImages event, Emitter<AnalyzeState> emit) {
    emit(const AnalyzeInitial());
  }
  
  Future<void> _onStartAnalysis(StartAnalysis event, Emitter<AnalyzeState> emit) async {
    if (!state.canStartAnalysis) {
      emit(AnalyzeError(
        images: state.images,
        message: 'Please add at least 3 images to start analysis',
      ));
      return;
    }
    
    emit(AnalyzeLoading(
      images: state.images,
      message: 'Uploading images...',
    ));
    
    try {
      emit(AnalyzeLoading(
        images: state.images,
        message: 'Analyzing plant health...',
      ));
      
      final result = await _repository.analyzePlant(
        images: state.images,
        plantType: event.plantType,
      );
      
      emit(AnalyzeSuccess(images: state.images, result: result));
    } catch (e) {
      emit(AnalyzeError(
        images: state.images,
        message: e.toString(),
      ));
    }
  }
  
  void _onResetAnalysis(ResetAnalysis event, Emitter<AnalyzeState> emit) {
    emit(const AnalyzeInitial());
  }
}
