<?php

use App\Http\Controllers\Api\AnalyzeController;
use App\Http\Controllers\Api\HistoryController;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| SmartPlant Vision API endpoints for plant health analysis.
|
*/

// Plant Analysis
Route::post('/analyze', [AnalyzeController::class, 'analyze']);

// History
Route::get('/history', [HistoryController::class, 'index']);
Route::get('/history/{id}', [HistoryController::class, 'show']);

// Health check
Route::get('/health', function () {
    return response()->json([
        'status' => 'ok',
        'version' => '1.0.0',
        'timestamp' => now()->toIso8601String(),
    ]);
});
