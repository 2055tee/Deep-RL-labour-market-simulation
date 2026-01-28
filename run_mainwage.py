from min_wage_model import LaborMarketModel
from min_wage_model import Worker, Firm
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np # import numpy for plotting averages

# Set a style for better looking plots
plt.style.use('ggplot') 

# --- FUNCTIONS FOR PLOTTING IN FINAL STEP ---

def plot_employment_rate(df, min_wage_param_list):
    plt.figure(figsize=(10, 6))
    # Aggregate data: Calculate mean employment rate over the last few steps for each min_wage
    # Or, just use the final step as in your original script
    
    # We will plot the final Employment Rate vs Min Wage
    plt.plot(df['min_wage'], df['employment_rate'], marker='o')
    plt.title('Employment Rate vs. Starting Minimum Wage (Final Step)')
    plt.xlabel('Starting Minimum Wage')
    plt.ylabel('Employment Rate')
    plt.grid(True)
    plt.show()
    

def plot_wage_distribution(df, min_wage_param_list):
    # Select all min_wage values to plot
    selected_wages = min_wage_param_list
    
    plt.figure(figsize=(12, 8))
    for mw in selected_wages:
        # Filter the final results DataFrame for the specific min_wage
        wages_data = df[df['min_wage'] == mw]['final_wages'].iloc[0]
        if wages_data:
             # wages_data is a list of wages, plot its distribution using a histogram
             plt.hist(wages_data, bins=15, alpha=0.6, label=f'Min Wage: {mw}')
    
    plt.title('Wage Distribution at Final Step for Selected Starting Minimum Wages')
    plt.xlabel('Monthly Wage')
    plt.ylabel('Frequency (Number of Workers)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# all worker wages at final step in histogram
def plot_worker_wage_distribution(df, min_wage):
    # set window size
    plt.figure(figsize=(12, 8))

    # get the wages data for the specific min_wage
    wages_data = df[df['min_wage'] == min_wage]['final_wages'].iloc[0]
    if wages_data:
         # wages_data is a list of wages, plot its distribution using a histogram
         plt.hist(wages_data, bins=15, alpha=0.7, color='blue')
    
    plt.title(f'Wage Distribution at Final Step for Starting Minimum Wage: {min_wage}')
    plt.xlabel('Monthly Wage')
    plt.ylabel('Frequency (Number of Workers)')
    plt.grid(True)
    plt.show()
    

def plot_firm_metrics(df, min_wage_param_list):
    # Average Capital
    plt.plot(df['min_wage'], df['avg_capital'], marker='o', color='green')
    plt.title('Average Firm Capital vs. Starting Minimum Wage (Final Step)')
    plt.xlabel('Starting Minimum Wage')
    plt.ylabel('Average Capital')
    plt.grid(True)
    plt.show()

def plot_firm_profit_distribution(df, min_wage):
    # set window size
    plt.figure(figsize=(12, 8))

    # Average Profit
    plt.plot(df['min_wage'], df['avg_profit'], marker='o', color='orange')
    plt.title('Average Firm Profit vs. Starting Minimum Wage (Final Step)')
    plt.xlabel('Starting Minimum Wage')
    plt.ylabel('Average Profit')
    plt.grid(True)
    plt.show()

def plot_firm_size(df, min_wage_param_list):
    # Firm Size (Average Number of Workers per Firm)
    plt.figure(figsize=(10, 6))
    plt.plot(df['min_wage'], df['avg_firm_size'], marker='o', color='purple')
    plt.title('Average Firm Size vs. Starting Minimum Wage (Final Step)')
    plt.xlabel('Starting Minimum Wage')
    plt.ylabel('Average Number of Workers per Firm')
    plt.grid(True)
    plt.show()
    
# --- FUNCTIONS FOR PLOTTING IN ALL STEP---
def plot_employment_vs_profit(df=None, min_wage_param_list=None):
    # Example: suppose each simulation run stores summary metrics
    # Columns: ['min_wage', 'employment_rate', 'avg_firm_profit']
    # test data
    summary_data = df.groupby('min_wage').agg({
        'EmploymentRate': 'last',
        'AverageProfit': 'last'
    }).reset_index()
    
    # summary_data = pd.DataFrame({
    #     'min_wage': [300, 400, 500, 600, 700],
    #     'EmploymentRate': [0.92, 0.88, 0.83, 0.76, 0.68],
    #     'AverageProfit': [1200, 1000, 780, 560, 400]
    # })
    # summary_data = pd

    # Plot trade-off curve
    plt.figure(figsize=(7, 5))
    plt.plot(summary_data['EmploymentRate'], summary_data['AverageProfit'],
            marker='o', linestyle='-', color='steelblue')

    # Annotate each point with its corresponding minimum wage
    for i, wage in enumerate(summary_data['min_wage']):
        plt.text(summary_data['EmploymentRate'][i] + 0.002,
                summary_data['AverageProfit'][i] + 10,
                f"{wage}", fontsize=8, color='darkslategray')

    plt.title("Trade-off Between Employment and Profit Across Wage Levels", fontsize=13)
    plt.xlabel("Employment Rate")
    plt.ylabel("Average Firm Profit")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    

# --- MAIN SIMULATION LOOP ---

results = []
all_data = []
min_wage_param_list = range(300, 601, 50)

for min_wage in min_wage_param_list:
    # Use a fixed seed for better comparison across different min_wage parameters
    # The 'seed' argument to LaborMarketModel will handle the model's random state
    model = LaborMarketModel(N_workers=100, N_firms=10, min_wage=min_wage, seed=42) 
    
    for i in range(60):  # 24 steps per simulation
        # print(f"************************Step {i+1} for min_wage {min_wage}************************")
        model.step()
        
        # collect all data at each step
        data = model.datacollector.get_model_vars_dataframe()
        data['min_wage'] = min_wage
        data['step'] = i + 1
        all_data.append(data)
        
        
        
        
    data = model.datacollector.get_model_vars_dataframe()
    final = data.iloc[-1]
    
    # Collect all the final model-level metrics and the collected lists
    results.append({
        "min_wage": min_wage,
        "employment_rate": final["EmploymentRate"],
        "avg_wage": final["AverageWage"],
        "avg_profit": final["AverageProfit"],
        "avg_firm_size": final["AvgFirmSize"],
        "avg_capital": final["AvgFirmCapital"],
        "final_wages": final["AllEmployedWages"], # List of all employed worker wages at the final step
        "final_profits": final["AllFirmProfits"], # List of all firm profits at the final step
        "final_capitals": final["AllFirmCapitals"], # List of all firm capitals at the final step
        "final_firm_sizes": final["AllFirmSizes"], # List of all firm sizes at the final step
    })

df = pd.DataFrame(results)
print("\n--- Final Results per Minimum Wage Parameter ---\n")
print(df[['min_wage', 'employment_rate', 'avg_wage', 'avg_profit', 'avg_firm_size', 'avg_capital']])

# --- GENERATE GRAPHS ---
print("\n--- Generating Graphs ---\n")

# 1. Employment Rate
plot_employment_rate(df, min_wage_param_list)

# 2. Wage Distribution (for selected min_wage values)
plot_wage_distribution(df, min_wage_param_list)

plot_firm_profit_distribution(df, min_wage_param_list)

# 3. Firm Profit and Capital (Average)
plot_firm_metrics(df, min_wage_param_list)

# 4. Firm Size (Average)
plot_firm_size(df, min_wage_param_list)

# # 5. Worker Wage Distribution for a specific min_wage
# # Tobe use when calibrated
# plot_worker_wage_distribution(df, min_wage=300)
# plot_worker_wage_distribution(df, min_wage=600)

# # Combine all collected data into a single DataFrame
# all_data_df = pd.concat(all_data, ignore_index=True)
# all_data_df.to_csv("min_wage_simulation_data.csv", index=False)

# # Plot Employment Rate vs Firm Profit over all steps
# plot_employment_vs_profit(all_data_df)