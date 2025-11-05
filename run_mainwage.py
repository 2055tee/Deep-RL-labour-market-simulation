from min_wage_model import LaborMarketModel
from min_wage_model import Worker, Firm
import pandas as pd

results = []

for min_wage in range(300, 601, 50):
    model = LaborMarketModel(N_workers=100, N_firms=10, min_wage=min_wage)
    for i in range(24):  # 24 steps per simulation (e.g., 2 years since each step is a month)
        print(f"************************Step {i+1} for min_wage {min_wage}************************")
        model.step()
       
        
    data = model.datacollector.get_model_vars_dataframe()
    final = data.iloc[-1]
    results.append({
        "min_wage": min_wage,
        "employment_rate": final["EmploymentRate"],
        "avg_wage": final["AverageWage"],
        "avg_profit": final["AverageProfit"]
    })

df = pd.DataFrame(results)
print(df)