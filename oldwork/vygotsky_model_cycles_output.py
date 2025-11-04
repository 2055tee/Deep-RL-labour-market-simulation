from mesa_vygotsky_model import TutoringModel, get_avg_knowledge, get_avg_zpd_size, get_avg_happiness

num_students = 25
steps_to_run = 100

model = TutoringModel(num_students)

with open('vygotsky_model_cycles_output.txt', 'w') as f:
    for i in range(steps_to_run):
        if model.running:
            model.step()
            f.write(f"--- Step {i+1} ---\n")
            f.write(f"Avg Knowledge: {get_avg_knowledge(model):.2f}, Avg ZPD Size: {get_avg_zpd_size(model):.2f}, Avg Happiness: {get_avg_happiness(model):.2f}\n")
            for agent in model.schedule.agents:
                f.write(f"Agent {agent.unique_id}: K={agent.current_knowledge:.2f}, P={agent.potential_knowledge:.2f}, H={agent.happiness:.2f}\n")
            f.write("\n")
        else:
            f.write(f"Simulation ended early at step {i+1} due to high average knowledge.\n")
            break
print("Full cycle output written to vygotsky_model_cycles_output.txt")
