<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\ScanImage;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class PlantAnalyzerService
{
    /**
     * Path to Python executable
     */
    protected string $pythonPath;

    /**
     * Path to the analysis script
     */
    protected string $scriptPath;

    /**
     * Base path for smartplant scripts
     */
    protected string $smartplantPath;

    public function __construct()
    {
        $this->pythonPath = config('services.smartplant.python_path', 'python');
        $this->smartplantPath = config('services.smartplant.base_path', base_path('../'));
        $this->scriptPath = $this->smartplantPath . '/smartplant_enhanced.py';
    }

    /**
     * Analyze a set of plant images.
     *
     * @param Scan $scan The scan record
     * @param array $imagePaths Array of uploaded image paths
     * @return array Analysis results
     */
    public function analyze(Scan $scan, array $imagePaths): array
    {
        // Create temporary folder for this analysis
        $tempFolder = storage_path("app/temp/scan_{$scan->id}");
        $outputDir = storage_path("app/public/scans/{$scan->id}");

        // Ensure directories exist
        if (!is_dir($tempFolder)) {
            mkdir($tempFolder, 0755, true);
        }
        if (!is_dir($outputDir)) {
            mkdir($outputDir, 0755, true);
        }

        // Copy images to temp folder
        foreach ($imagePaths as $index => $path) {
            $filename = basename($path);
            copy(storage_path("app/{$path}"), "{$tempFolder}/{$filename}");
        }

        // Run the Python analysis script
        $result = $this->runAnalysis($tempFolder, $outputDir);

        // Clean up temp folder
        $this->cleanupTempFolder($tempFolder);

        if ($result['status'] === 'success') {
            // Process and store results
            $this->storeResults($scan, $result, $outputDir);
        }

        return $result;
    }

    /**
     * Run the Python analysis script.
     */
    protected function runAnalysis(string $inputFolder, string $outputDir): array
    {
        // Build command arguments
        $scriptContent = $this->generateAnalysisScript($inputFolder, $outputDir);
        $tempScript = storage_path('app/temp/run_analysis.py');
        file_put_contents($tempScript, $scriptContent);

        $process = new Process([
            $this->pythonPath,
            $tempScript
        ]);

        $process->setTimeout(300); // 5 minutes timeout
        $process->setWorkingDirectory($this->smartplantPath);

        try {
            $process->mustRun();
            $output = $process->getOutput();

            // Parse JSON result from the script output
            $jsonStart = strpos($output, '{"status"');
            if ($jsonStart !== false) {
                $jsonStr = substr($output, $jsonStart);
                $result = json_decode($jsonStr, true);
                if ($result) {
                    return $result;
                }
            }

            // Try to read from output file
            $reportPath = "{$outputDir}/analysis_report.json";
            if (file_exists($reportPath)) {
                $result = json_decode(file_get_contents($reportPath), true);
                if ($result) {
                    return $result;
                }
            }

            return [
                'status' => 'error',
                'message' => 'Failed to parse analysis results'
            ];

        } catch (ProcessFailedException $e) {
            Log::error('Plant analysis failed', [
                'error' => $e->getMessage(),
                'output' => $process->getErrorOutput()
            ]);

            return [
                'status' => 'error',
                'message' => 'Analysis process failed: ' . $e->getMessage()
            ];
        } finally {
            // Clean up temp script
            if (file_exists($tempScript)) {
                unlink($tempScript);
            }
        }
    }

    /**
     * Generate a temporary Python script that calls the analyzer.
     */
    protected function generateAnalysisScript(string $inputFolder, string $outputDir): string
    {
        $inputFolder = str_replace('\\', '/', $inputFolder);
        $outputDir = str_replace('\\', '/', $outputDir);
        $smartplantPath = str_replace('\\', '/', $this->smartplantPath);

        return <<<PYTHON
import sys
import os
import json

# Add smartplant path to sys.path
sys.path.insert(0, '{$smartplantPath}')

# Change to smartplant directory
os.chdir('{$smartplantPath}')

# Import after path setup
from smartplant_enhanced import analyze_plant

# Run analysis
result = analyze_plant('{$inputFolder}', '{$outputDir}')

# Output as JSON
print(json.dumps(result))
PYTHON;
    }

    /**
     * Store analysis results in the database.
     */
    protected function storeResults(Scan $scan, array $result, string $outputDir): void
    {
        $summary = $result['plant_summary'] ?? [];
        $classification = $summary['classification'] ?? [];

        // Update scan record
        $scan->update([
            'status' => 'completed',
            'health_score' => $summary['health_score'] ?? null,
            'condition' => $summary['condition'] ?? null,
            'predicted_class' => $classification['predicted_class'] ?? null,
            'confidence' => $classification['confidence'] ?? null,
            'class_probabilities' => $classification['class_probabilities'] ?? null,
            'avg_lesion_area_percent' => $summary['avg_lesion_area_percent'] ?? null,
            'total_lesion_count' => $summary['total_lesion_count'] ?? null,
            'interpretation' => $result['interpretation'] ?? null,
        ]);

        // Store individual leaf results
        $leafResults = $result['leaf_results'] ?? [];
        $overlayImages = $result['overlay_images'] ?? [];

        foreach ($leafResults as $index => $leaf) {
            $classification = $leaf['classification'] ?? [];
            $lesionMetrics = $leaf['lesion_metrics'] ?? [];
            $veinMetrics = $leaf['vein_morphometry'] ?? [];

            // Get overlay path relative to storage
            $overlayPath = null;
            if (isset($overlayImages[$index])) {
                $overlayBasename = basename($overlayImages[$index]);
                $overlayPath = "public/scans/{$scan->id}/{$overlayBasename}";
            }

            ScanImage::create([
                'scan_id' => $scan->id,
                'leaf_index' => $leaf['leaf_index'] ?? ($index + 1),
                'filename' => $leaf['filename'] ?? "leaf_" . ($index + 1) . ".jpg",
                'original_path' => "public/scans/{$scan->id}/original_" . ($index + 1) . ".jpg",
                'overlay_path' => $overlayPath,
                'predicted_class' => $classification['predicted_class'] ?? null,
                'confidence' => $classification['confidence'] ?? null,
                'probabilities' => $classification['probabilities'] ?? null,
                'lesion_count' => $lesionMetrics['lesion_count'] ?? 0,
                'lesion_area_percent' => $lesionMetrics['lesion_area_percent'] ?? 0,
                'vein_length_px' => $veinMetrics['vein_length_px'] ?? null,
                'vein_density_percent' => $veinMetrics['vein_density_percent'] ?? null,
                'vein_continuity' => $veinMetrics['vein_continuity'] ?? null,
            ]);
        }
    }

    /**
     * Clean up temporary folder.
     */
    protected function cleanupTempFolder(string $folder): void
    {
        if (!is_dir($folder)) {
            return;
        }

        $files = glob("{$folder}/*");
        foreach ($files as $file) {
            if (is_file($file)) {
                unlink($file);
            }
        }
        rmdir($folder);
    }
}
