<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\AnalyzeRequest;
use App\Models\Scan;
use App\Services\PlantAnalyzerService;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Storage;

class AnalyzeController extends Controller
{
    protected PlantAnalyzerService $analyzerService;

    public function __construct(PlantAnalyzerService $analyzerService)
    {
        $this->analyzerService = $analyzerService;
    }

    /**
     * Analyze plant images.
     *
     * POST /api/analyze
     */
    public function analyze(AnalyzeRequest $request): JsonResponse
    {
        // Create scan record
        $scan = Scan::create([
            'user_id' => $request->user()?->id,
            'plant_type' => $request->input('plant_type', 'rice'),
            'status' => 'processing',
        ]);

        // Store uploaded images
        $imagePaths = [];
        $scanFolder = "public/scans/{$scan->id}";
        Storage::makeDirectory($scanFolder);

        foreach ($request->file('images') as $index => $image) {
            $filename = "original_" . ($index + 1) . "." . $image->getClientOriginalExtension();
            $path = $image->storeAs($scanFolder, $filename);
            $imagePaths[] = $path;
        }

        try {
            // Run analysis
            $result = $this->analyzerService->analyze($scan, $imagePaths);

            if ($result['status'] === 'error') {
                $scan->update(['status' => 'failed']);
                return response()->json([
                    'status' => 'error',
                    'message' => $result['message'] ?? 'Analysis failed',
                ], 500);
            }

            // Reload scan with images
            $scan->refresh();
            $scan->load('images');

            return response()->json($this->formatResponse($scan));

        } catch (\Exception $e) {
            $scan->update(['status' => 'failed']);
            return response()->json([
                'status' => 'error',
                'message' => 'An error occurred during analysis: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Format the scan response.
     */
    protected function formatResponse(Scan $scan): array
    {
        $leaves = $scan->images->map(function ($image) {
            return [
                'leaf_index' => $image->leaf_index,
                'filename' => $image->filename,
                'predicted_class' => $image->predicted_class,
                'confidence' => round($image->confidence, 3),
                'lesion_count' => $image->lesion_count,
                'lesion_area_percent' => round($image->lesion_area_percent, 2),
                'vein_status' => $image->vein_status,
                'vein_metrics' => [
                    'length_px' => $image->vein_length_px,
                    'density_percent' => round($image->vein_density_percent, 2),
                    'continuity' => round($image->vein_continuity, 3),
                ],
                'original_url' => $image->original_url,
                'overlay_url' => $image->overlay_url,
            ];
        });

        return [
            'status' => 'success',
            'scan_id' => $scan->id,
            'timestamp' => $scan->created_at->toIso8601String(),
            'plant_type' => $scan->plant_type,
            'summary' => [
                'health_score' => round($scan->health_score, 1),
                'condition' => $scan->condition,
                'predicted_class' => $scan->predicted_class,
                'confidence' => round($scan->confidence, 3),
                'class_probabilities' => $scan->class_probabilities,
                'avg_lesion_area_percent' => round($scan->avg_lesion_area_percent, 2),
                'total_lesion_count' => $scan->total_lesion_count,
            ],
            'leaves' => $leaves,
            'interpretation' => $scan->interpretation ?? [
                'classification_note' => 'Based solely on CNN deep learning analysis',
                'health_score_note' => 'Heuristic indicator (0-100), not a biological diagnosis',
                'vein_note' => 'Vein analysis is for visual support only, not diagnostic',
            ],
        ];
    }
}
