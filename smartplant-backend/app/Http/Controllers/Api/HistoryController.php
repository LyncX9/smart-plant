<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Scan;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class HistoryController extends Controller
{
    /**
     * List all scans with filtering.
     *
     * GET /api/history
     */
    public function index(Request $request): JsonResponse
    {
        $query = Scan::query()
            ->with('images')
            ->where('status', 'completed')
            ->orderBy('created_at', 'desc');

        // Apply filters
        if ($request->filled('plant_type')) {
            $query->ofPlantType($request->plant_type);
        }

        if ($request->filled('condition')) {
            $query->withCondition($request->condition);
        }

        if ($request->filled('date_from') || $request->filled('date_to')) {
            $query->dateBetween($request->date_from, $request->date_to);
        }

        // Paginate
        $perPage = min($request->input('per_page', 15), 50);
        $scans = $query->paginate($perPage);

        return response()->json([
            'data' => $scans->map(function ($scan) {
                return $this->formatListItem($scan);
            }),
            'meta' => [
                'current_page' => $scans->currentPage(),
                'last_page' => $scans->lastPage(),
                'per_page' => $scans->perPage(),
                'total' => $scans->total(),
            ],
        ]);
    }

    /**
     * Get a single scan detail.
     *
     * GET /api/history/{id}
     */
    public function show(int $id): JsonResponse
    {
        $scan = Scan::with('images')->findOrFail($id);

        return response()->json([
            'data' => $this->formatDetail($scan),
        ]);
    }

    /**
     * Format scan for list view.
     */
    protected function formatListItem(Scan $scan): array
    {
        return [
            'id' => $scan->id,
            'plant_type' => $scan->plant_type,
            'health_score' => round($scan->health_score, 1),
            'condition' => $scan->condition,
            'predicted_class' => $scan->predicted_class,
            'confidence' => round($scan->confidence, 3),
            'leaves_count' => $scan->images->count(),
            'created_at' => $scan->created_at->toIso8601String(),
        ];
    }

    /**
     * Format scan for detail view.
     */
    protected function formatDetail(Scan $scan): array
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
            'id' => $scan->id,
            'timestamp' => $scan->created_at->toIso8601String(),
            'plant_type' => $scan->plant_type,
            'status' => $scan->status,
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
