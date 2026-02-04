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
        Schema::create('scans', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->nullable()->constrained()->onDelete('cascade');
            $table->string('plant_type')->default('rice');
            $table->enum('status', ['pending', 'processing', 'completed', 'failed'])->default('pending');
            $table->decimal('health_score', 5, 2)->nullable();
            $table->enum('condition', ['Healthy', 'Diseased', 'Uncertain'])->nullable();
            $table->string('predicted_class')->nullable();
            $table->decimal('confidence', 5, 4)->nullable();
            $table->json('class_probabilities')->nullable();
            $table->decimal('avg_lesion_area_percent', 5, 2)->nullable();
            $table->integer('total_lesion_count')->nullable();
            $table->json('interpretation')->nullable();
            $table->timestamps();
            
            // Indexes for filtering
            $table->index('plant_type');
            $table->index('condition');
            $table->index('created_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scans');
    }
};
