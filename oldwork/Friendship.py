from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

class PersonAgent(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None   # required for MultiGrid
        self.beh_pred = random.uniform(0, model.beh_pred_range)
        self.happiness = model.starting_happiness
        self.alive = True
        self.friends = {}

    def step(self):
        if not self.alive:
            return
        self.making_friends()
        self.friendship_ending()
        self.fighting()
        self.check_bodycount()

    def making_friends(self):
        others = [a for a in self.model.schedule.agents if a.unique_id != self.unique_id and a.alive]
        if others:
            target = random.choice(others)
            dist_target = abs(target.beh_pred - self.beh_pred)
            if dist_target < self.model.max_distance_for_friendship:
                # Create friendship (bidirectional)
                weight = self.model.max_distance_for_friendship - dist_target
                self.friends[target.unique_id] = {"weight": weight, "expiry": self.model.expiration_time}
                target.friends[self.unique_id] = {"weight": weight, "expiry": self.model.expiration_time}
                # Increase happiness for both
                self.happiness += self.model.increase_if_friendship
                target.happiness += self.model.increase_if_friendship

    def friendship_ending(self):
        expired = []
        for fid, rel in self.friends.items():
            rel["expiry"] -= 1
            if rel["expiry"] <= 0:
                expired.append(fid)
        for fid in expired:
            del self.friends[fid]
            if fid in self.model.schedule._agents:
                other = self.model.schedule._agents[fid]
                if self.unique_id in other.friends:
                    del other.friends[self.unique_id]

    def fighting(self):
        for fid in list(self.friends.keys()):
            if random.random() < self.model.prob_fight:
                # Both agents lose happiness
                other = self.model.schedule._agents[fid]
                self.happiness -= self.model.loss_if_fight
                other.happiness -= self.model.loss_if_fight
                # Remove friendship
                del self.friends[fid]
                if self.unique_id in other.friends:
                    del other.friends[self.unique_id]

    def check_bodycount(self):
        if self.happiness <= 0 and self.alive:
            self.alive = False
            # Lose all friendships, decrease happiness of connected agents
            for fid in list(self.friends.keys()):
                other = self.model.schedule._agents[fid]
                other.happiness -= self.model.loss_if_fight
                other.friends.pop(self.unique_id, None)  # safe removal
            self.friends.clear()



class FriendshipModel(Model):
    def __init__(self, number_of_people=20, starting_happiness=7, increase_if_friendship=0.8,
                 beh_pred_range=10, max_distance_for_friendship=5, expiration_time=10,
                 prob_fight=0.1, loss_if_fight=1):
        super().__init__()
        self.num_agents = number_of_people
        self.starting_happiness = starting_happiness
        self.increase_if_friendship = increase_if_friendship
        self.beh_pred_range = beh_pred_range
        self.max_distance_for_friendship = max_distance_for_friendship
        self.expiration_time = expiration_time
        self.prob_fight = prob_fight
        self.loss_if_fight = loss_if_fight

        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(20, 20, True)

        for i in range(self.num_agents):
            a = PersonAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Happiness": "happiness", "Alive": "alive"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# Example run
if __name__ == "__main__":
    model = FriendshipModel()
    for i in range(50):
        model.step()
    data = model.datacollector.get_agent_vars_dataframe()
    print(data.tail())
