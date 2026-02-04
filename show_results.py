import json

with open("model_comparison.json", "r") as f:
    data = json.load(f)

print("=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)
print(f"{'Rank':<6}{'Model':<30}{'Confidence':<15}{'Healthy':<10}")
print("-" * 70)

for r in data["ranking"]:
    print(f"{r['rank']:<6}{r['model']:<30}{r['avg_confidence_percent']:<15.1f}{r['healthy_detected']}/{r['total_images']}")

print("=" * 70)
print(f"BEST MODEL: {data['best_model']}")
print("=" * 70)
