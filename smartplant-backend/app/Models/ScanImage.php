<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\Storage;

class ScanImage extends Model
{
    use HasFactory;

    protected $fillable = [
        'scan_id',
        'leaf_index',
        'filename',
        'original_path',
        'overlay_path',
        'predicted_class',
        'confidence',
        'probabilities',
        'lesion_count',
        'lesion_area_percent',
        'vein_length_px',
        'vein_density_percent',
        'vein_continuity',
    ];

    protected $casts = [
        'confidence' => 'decimal:4',
        'probabilities' => 'array',
        'lesion_area_percent' => 'decimal:2',
        'vein_density_percent' => 'decimal:2',
        'vein_continuity' => 'decimal:3',
    ];

    /**
     * Get the scan that owns this image.
     */
    public function scan(): BelongsTo
    {
        return $this->belongsTo(Scan::class);
    }

    /**
     * Get the URL for the original image.
     */
    public function getOriginalUrlAttribute(): ?string
    {
        if (!$this->original_path) {
            return null;
        }
        return url(Storage::url($this->original_path));
    }

    /**
     * Get the URL for the overlay image.
     */
    public function getOverlayUrlAttribute(): ?string
    {
        if (!$this->overlay_path) {
            return null;
        }
        return url(Storage::url($this->overlay_path));
    }

    /**
     * Determine vein status based on metrics.
     * This is for display purposes only - NOT diagnostic.
     */
    public function getVeinStatusAttribute(): string
    {
        if ($this->vein_density_percent === null) {
            return 'Unknown';
        }
        
        // Good: density > 15% and continuity > 0.2
        if ($this->vein_density_percent > 15 && $this->vein_continuity > 0.2) {
            return 'Good';
        }
        
        return 'Irregular';
    }
}
