# train_coop_then_comp.py
# Runs cooperative training then competitive training back-to-back.
# Launch this after solo training finishes.

import subprocess, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

for scenario in ["cooperative", "competitive"]:
    print(f"\n{'='*60}")
    print(f"  Starting {scenario} training...")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=os.path.join(ROOT, scenario),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {scenario} training failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  {scenario} training done.")

print("\nAll training complete. Run: python compare_all.py")
