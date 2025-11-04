from Friendship import FriendshipModel

# Run the model and write all agent states to a file after each cycle
model = FriendshipModel()
num_steps = 50

with open('friendship_cycles_output.txt', 'w') as f:
    for i in range(num_steps):
        f.write(f"--- Step {i+1} ---\n")
        model.step()
        for agent in model.schedule.agents:
            f.write(f"Agent {agent.unique_id}: happiness={agent.happiness:.2f}, alive={agent.alive}, friends={list(agent.friends.keys())}\n")
        f.write("\n")
print("Full cycle output written to friendship_cycles_output.txt")
