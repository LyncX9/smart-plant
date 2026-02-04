# SmartPlant Vision - Flutter App

An AI-powered plant health analysis application built with Flutter and Laravel.

## Project Structure

```
smartplant-app/
├── lib/
│   ├── core/
│   │   ├── constants/
│   │   │   ├── api_endpoints.dart    # API configuration
│   │   │   └── colors.dart           # App color palette
│   │   └── theme/
│   │       └── app_theme.dart        # Material theme
│   ├── data/
│   │   ├── datasources/
│   │   │   └── plant_api_datasource.dart  # HTTP client
│   │   ├── models/
│   │   │   ├── leaf_result_model.dart
│   │   │   └── scan_model.dart
│   │   └── repositories/
│   │       └── plant_repository_impl.dart
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── leaf_result.dart
│   │   │   └── scan.dart
│   │   └── repositories/
│   │       └── plant_repository.dart
│   ├── presentation/
│   │   ├── bloc/
│   │   │   ├── analyze/
│   │   │   │   └── analyze_bloc.dart
│   │   │   └── history/
│   │   │       └── history_bloc.dart
│   │   ├── screens/
│   │   │   ├── analyze/
│   │   │   │   ├── capture_screen.dart
│   │   │   │   └── result_screen.dart
│   │   │   ├── history/
│   │   │   │   ├── history_screen.dart
│   │   │   │   └── history_detail_screen.dart
│   │   │   ├── home_screen.dart
│   │   │   └── splash_screen.dart
│   │   └── widgets/
│   │       ├── health_score_gauge.dart
│   │       └── status_badge.dart
│   ├── injection.dart                 # Dependency injection
│   └── main.dart                      # App entry point
├── assets/
│   ├── fonts/                         # Outfit font files
│   ├── icons/                         # App icons
│   └── images/                        # App images
└── pubspec.yaml                       # Dependencies
```

## Features

- **AI-Powered Analysis**: Upload 3-7 leaf images for disease detection
- **Health Score**: Visual gauge showing plant health (0-100)
- **Multi-Leaf Support**: Individual analysis for each leaf
- **Vein Morphometry**: Visual vein analysis display (non-diagnostic)
- **History & Filtering**: View past scans with date/condition filters

## Setup

### Prerequisites
- Flutter SDK 3.0+
- Android Studio / VS Code
- Laravel backend running

### Installation

1. Install dependencies:
```bash
flutter pub get
```

2. Update API URL in `lib/core/constants/api_endpoints.dart`:
```dart
static const String baseUrl = 'http://YOUR_API_URL/api';
```

3. Run the app:
```bash
flutter run
```

## Backend API

This app connects to a Laravel backend. Required endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Upload images for analysis |
| GET | `/api/history` | List scan history |
| GET | `/api/history/{id}` | Get scan details |

## Design System

### Colors
- **Primary Green**: `#2ECC71`
- **Accent Green**: `#00E676`
- **Healthy**: `#4CAF50`
- **Warning**: `#FFC107`
- **Error**: `#F44336`

### Typography
- **Font**: Outfit (Google Fonts)
- Clean, modern agricultural-tech aesthetic
