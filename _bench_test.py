
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Override constants before they're used
import compare_all
compare_all.N_SEEDS = 1
compare_all.CONFIGS = [compare_all.CONFIGS[0]]  # small only

print("Test mode: 1 seed, small market only")
coop_res = compare_all.run_cooperative()
if coop_res:
    print("Coop results OK, tags:", list(coop_res.keys()))
    compare_all.chart_profit(coop_res, compare_all.BENCH / "cooperative", "Cooperative", compare_all.AI_COL["cooperative"])
    compare_all.chart_scorecard(coop_res, compare_all.BENCH / "cooperative", "Cooperative", compare_all.AI_COL["cooperative"])
    print("Charts OK")
else:
    print("Coop skipped")
