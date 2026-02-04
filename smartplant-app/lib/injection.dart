import 'package:get_it/get_it.dart';

import 'domain/repositories/plant_repository.dart';
import 'data/repositories/plant_repository_impl.dart';
import 'data/datasources/plant_api_datasource.dart';

final getIt = GetIt.instance;

Future<void> configureDependencies() async {
  // Data sources
  getIt.registerLazySingleton<PlantApiDataSource>(
    () => PlantApiDataSource(),
  );

  // Repositories
  getIt.registerLazySingleton<PlantRepository>(
    () => PlantRepositoryImpl(dataSource: getIt<PlantApiDataSource>()),
  );
}
