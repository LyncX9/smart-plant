<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class Scan extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'plant_type',
        'status',
        'health_score',
        'condition',
        'predicted_class',
        'confidence',
        'class_probabilities',
        'avg_lesion_area_percent',
        'total_lesion_count',
        'interpretation',
    ];

    protected $casts = [
        'health_score' => 'decimal:2',
        'confidence' => 'decimal:4',
        'class_probabilities' => 'array',
        'avg_lesion_area_percent' => 'decimal:2',
        'interpretation' => 'array',
    ];

    /**
     * Get the images for this scan.
     */
    public function images(): HasMany
    {
        return $this->hasMany(ScanImage::class)->orderBy('leaf_index');
    }

    /**
     * Get the user that owns this scan.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Scope for filtering by plant type.
     */
    public function scopeOfPlantType($query, string $plantType)
    {
        return $query->where('plant_type', $plantType);
    }

    /**
     * Scope for filtering by condition.
     */
    public function scopeWithCondition($query, string $condition)
    {
        return $query->where('condition', $condition);
    }

    /**
     * Scope for filtering by date range.
     */
    public function scopeDateBetween($query, $from, $to)
    {
        if ($from) {
            $query->whereDate('created_at', '>=', $from);
        }
        if ($to) {
            $query->whereDate('created_at', '<=', $to);
        }
        return $query;
    }

    /**
     * Get storage path for this scan's files.
     */
    public function getStoragePath(): string
    {
        return "scans/{$this->id}";
    }
}
