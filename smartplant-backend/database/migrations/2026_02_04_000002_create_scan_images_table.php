<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('scan_images', function (Blueprint $table) {
            $table->id();
            $table->foreignId('scan_id')->constrained()->onDelete('cascade');
            $table->integer('leaf_index');
            $table->string('filename');
            $table->string('original_path');
            $table->string('overlay_path')->nullable();
            
            // Classification results
            $table->string('predicted_class')->nullable();
            $table->decimal('confidence', 5, 4)->nullable();
            $table->json('probabilities')->nullable();
            
            // Lesion metrics
            $table->integer('lesion_count')->default(0);
            $table->decimal('lesion_area_percent', 5, 2)->default(0);
            
            // Vein morphometry (visual support only)
            $table->integer('vein_length_px')->nullable();
            $table->decimal('vein_density_percent', 5, 2)->nullable();
            $table->decimal('vein_continuity', 5, 3)->nullable();
            
            $table->timestamps();
            
            $table->index(['scan_id', 'leaf_index']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scan_images');
    }
};
