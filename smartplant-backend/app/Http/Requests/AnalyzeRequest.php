<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class AnalyzeRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     */
    public function rules(): array
    {
        return [
            'plant_type' => 'sometimes|string|max:50',
            'images' => 'required|array|min:3|max:7',
            'images.*' => 'required|image|mimes:jpeg,jpg,png,bmp|max:10240', // 10MB max per image
        ];
    }

    /**
     * Get custom messages for validator errors.
     */
    public function messages(): array
    {
        return [
            'images.required' => 'Please upload at least 3 leaf images for analysis.',
            'images.min' => 'Minimum 3 leaf images are required for reliable analysis.',
            'images.max' => 'Maximum 7 leaf images allowed per analysis.',
            'images.*.image' => 'Each file must be a valid image.',
            'images.*.mimes' => 'Images must be in JPEG, PNG, or BMP format.',
            'images.*.max' => 'Each image must not exceed 10MB.',
        ];
    }
}
