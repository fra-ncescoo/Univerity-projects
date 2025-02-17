#multiproduct-multimaterial sales and invenotry DQL control

#2 final products and 3 raw materials
#product 1 is made by 2 units of material 1 (m1), 1 of m2 and 3 of m3
#product 2 is made by 1 units of material 1 (m1), 2 of m2 and 2 of m3

#a ROL(s,S) reordering system for the raw materials is implemented and indeed
#each order restock the quantity of that material to its maximum stock level


import numpy as np  # to work with arrays
import torch        # to build and train deep q models
import torch.nn as nn
import torch.optim as optim
import random       # to introduce randomic processes
from collections import deque  # for append and pop from the memory
import matplotlib.pyplot as plt  # for plotting
from itertools import product  # for cartesian product

# Define the environment class to define transitions and rewards
class InventoryManagementEnv:
    # Any state is made by the combination of the stock of m1, m2, m3, p1, p2 and the backorders of p1 and p2
    def __init__(self, max_inventory_material=40, max_inventory_product=20, max_backorder=4,
                 cost_material=[0.2, 0.5, 0.1], cost_backorder=2, cost_holding=0.5, cost_order=1, recipes=None, cost_variation=0.1):
        self.max_inventory_material = max_inventory_material
        self.max_inventory_product = max_inventory_product
        self.max_backorder = max_backorder
        self.cost_material = cost_material
        self.cost_variation=cost_variation
        self.cost_backorder = cost_backorder
        self.cost_holding = cost_holding
        self.order_cost = cost_order  #Fixed cost for placing an order
        self.recipes = recipes if recipes is not None else [[2, 1, 3], [1, 2, 2]]  # Recipe for product 1 and 2 as a matrix
        self.state = None
        self._create_action_space()
        self.reset()

    def _create_action_space(self):
        # Actions: Sale price for product 1, Sale price for product 2
        # Replenish material i to its maximum stock level or not replenish it
        replenish_actions = [0, 1]
        sale_prices_1 = [2,9,10,12]
        sale_prices_2 = [2,9,10,12]
        production_1=[0,5,10]
        production_2=[0,5,10]
        self.actions = np.array(list(product(replenish_actions, replenish_actions, replenish_actions, sale_prices_1, sale_prices_2,production_1,production_2)))

    def reset(self):
        # Randomly initialize the state: inventory levels of m1, m2, m3, product1, product2, and backorders
        self.state = np.array([random.randint(0,self.max_inventory_material),  # m1
                               random.randint(0,self.max_inventory_material),  # m2
                               random.randint(0,self.max_inventory_material),  # m3
                               random.randint(0, self.max_inventory_product),   # product 1
                               random.randint(0, self.max_inventory_product),   # product 2
                               random.randint(0, self.max_backorder),           # backorder product 1
                               random.randint(0, self.max_backorder),           # backorder product 2
                               random.choice([2,5,9,10,12]),  # previous sale price 1
                               random.choice([2,5,9,10,12])])  # previous sale price 2
        return self.state

    def step(self, action):
        replenish_m1, replenish_m2, replenish_m3, sale_price_1, sale_price_2, prod_1,prod_2 = action

        # Extract values from the state array
        inventory_m1 = self.state[0]
        inventory_m2 = self.state[1]
        inventory_m3 = self.state[2]
        inventory_product_1 = self.state[3]
        inventory_product_2 = self.state[4]
        backorder_1 = self.state[5]
        backorder_2 = self.state[6]
        prev_sales_1 = self.state[7]
        prev_sales_2 = self.state[8]

        # Demand for each product based on the sale price and previous sales
        demand_1 = max(0, round(np.random.normal(15 - 0.8 * sale_price_1 - 0.8 * prev_sales_1, 1)))
        demand_2 = max(0, round(np.random.normal(14 - 0.7 * sale_price_2 - 0.4 * prev_sales_2, 1)))

        #production of p1
        possible_products_1 = min(inventory_m1 // self.recipes[0][0], inventory_m2 // self.recipes[0][1], inventory_m3 // self.recipes[0][2])
        production_1=min(prod_1,possible_products_1)
        inventory_m1 -= production_1 * self.recipes[0][0]
        inventory_m2 -= production_1 * self.recipes[0][1]
        inventory_m3 -= production_1 * self.recipes[0][2]
        #sales of p1
        sales_1=min(demand_1+backorder_1,inventory_product_1+production_1)
        
        #inventory overstock penality for p1
        inventory_penality_1=0
        if (inventory_product_1+production_1-sales_1)>self.max_inventory_product:
            inventory_penality_1=-10
        inventory_product_1=min(self.max_inventory_product, inventory_product_1+production_1-sales_1)
    
        #production of p2
        possible_products_2 = min(inventory_m1 // self.recipes[1][0], inventory_m2 // self.recipes[1][1], inventory_m3 // self.recipes[1][2])
        production_2 = min(prod_2, possible_products_2)
        inventory_m1 -= production_2 * self.recipes[1][0]
        inventory_m2 -= production_2 * self.recipes[1][1]
        inventory_m3 -= production_2 * self.recipes[1][2]
        #sales of p2
        sales_2 = min(demand_2+backorder_2, inventory_product_2+production_2)

        #inventory overstock penality for p2
        inventory_penality_2=0
        if (inventory_product_2+production_2-sales_2)>self.max_inventory_product:
            inventory_penality_2=-10
        inventory_product_2 = min(self.max_inventory_product,inventory_product_2+production_2-sales_2)

        #excess backorder penality
        backorder_penality_1=0
        backorder_penality_2=0
        if demand_1+backorder_1-sales_1>self.max_backorder and sales_1 < demand_1+backorder_1:
            backorder_penality_1=-10
        if demand_2+backorder_2-sales_2>self.max_backorder and sales_2<demand_2+backorder_2:
            backorder_penality_2=-10

        #Any remaining demand becomes backorder
        if sales_1 < demand_1+backorder_1:
            backorder_1 = min(self.max_backorder, (demand_1+backorder_1) - sales_1)
        else:
            backorder_1=0
        if sales_2 < demand_2+backorder_2:
            backorder_2 = min(self.max_backorder, (demand_2+backorder_2) - sales_2) 
        else:
            backorder_2=0
        

        # Update material inventory if replenish action is performed (with 5% chance of failure)
        replenished = False
        
        # Replenish material 1
        if replenish_m1 == 1:
            if random.random() > 0.005:  # 95% chance to successfully replenish
                inventory_m1 = self.max_inventory_material  # Replenish m1 to max
                replenished = True
            
        # Replenish material 2
        if replenish_m2 == 1:
            if random.random() > 0.005:  # 95% chance to successfully replenish
                inventory_m2 = self.max_inventory_material  # Replenish m2 to max
                replenished = True
            
        # Replenish material 3
        if replenish_m3 == 1:
            if random.random() > 0.005:  # 95% chance to successfully replenish
                inventory_m3 = self.max_inventory_material  # Replenish m3 to max
                replenished = True
            
        # Holding cost (cost of keeping a unit of m or p in inventory for a day)
        holding_cost = self.cost_holding * (inventory_m1 + inventory_m2 + inventory_m3 + inventory_product_1 + inventory_product_2)

        # Purchase cost (cost of replenished raw materials)
        self.cost_material=[base_cost*random.uniform(1-self.cost_variation,1+self.cost_variation) for base_cost in self.cost_material]
        purchase_cost = (self.max_inventory_material - inventory_m1) * self.cost_material[0] + \
                        (self.max_inventory_material - inventory_m2) * self.cost_material[1] + \
                        (self.max_inventory_material - inventory_m3) * self.cost_material[2]

        # Order cost (if any material was replenished)
        order_cost = self.order_cost if replenished else 0

        # Backorder cost
        backorder_cost = (backorder_1 + backorder_2) * self.cost_backorder

        # Reward calculation: sales revenue minus all costs and penalities
        reward = (sales_1 * sale_price_1) + (sales_2 * sale_price_2) - (holding_cost + purchase_cost + order_cost + backorder_cost)+(backorder_penality_1+backorder_penality_2)+(inventory_penality_1+inventory_penality_2)

        # Transition to next state
        next_state = np.array([inventory_m1, inventory_m2, inventory_m3, inventory_product_1, inventory_product_2, backorder_1, backorder_2, sales_1, sales_2])
        self.state = next_state

        return next_state, reward

    def get_state_size(self):
        return 9  # [inventory_m1, inventory_m2, inventory_m3, inventory_product_1, inventory_product_2, backorder_1, backorder_2, sales_1, sales_2]

    def get_action_size(self):
        return len(self.actions)  # Number of possible actions

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
gamma = 0.90
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.08
alpha = 0.005
episodes = 1000
days = 360
batch_size = 64
memory_size = 10000

# Initialize environment, Q-network, target network, and optimizer
env = InventoryManagementEnv()
state_size = env.get_state_size()
action_size = env.get_action_size()

q_network = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=alpha)

# Experience replay buffer
memory = deque(maxlen=memory_size)

# Training loop
cumulative_rewards = []

for episode in range(episodes):
    print(episode)
    state = env.reset()
    total_reward = 0

    for day in range(days):
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)

        # Epsilon-greedy action selection
        if random.random() < epsilon:                      #exploration
            action_idx = random.choice(range(action_size)) 
        else:
            with torch.no_grad():                          #exploitation
                q_values = q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()

        action = env.actions[action_idx]
        next_state, reward = env.step(action)

        memory.append((state, action_idx, reward, next_state))

        state = next_state
        total_reward += reward

        # Experience replay
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states = zip(*batch)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.int64)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)

            q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)    #pass the state through the network and gather the Q of the selected action
            next_q_values = target_network(next_states_tensor).max(1)[0]    #find the best possible reward for the next state in the batch
            target_q_values = rewards_tensor + gamma * next_q_values

            loss = nn.functional.mse_loss(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cumulative_rewards.append(total_reward)

    # Update target network
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Plot cumulative rewards
plt.plot(cumulative_rewards)
plt.title('Cumulative Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.show()

#optimal policy behaviour (from a random state under optimal policy)
optimal_policy = []

state = env.reset()  # Reset environment to the initial state

for day in range(days):
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)

    # Choose the best action using the trained Q-network (deterministic policy)
    with torch.no_grad():
        q_values = q_network(state_tensor)
        action_idx = torch.argmax(q_values).item()

    action = env.actions[action_idx]
    optimal_policy.append((state, action_idx, action))  # Store state, action index, and action taken

    next_state, reward = env.step(action)  # Apply the action to the environment
    state = next_state

# Display the optimal policy
for i, (s, action_idx, action) in enumerate(optimal_policy):
    print(f"Day {i+1}: State={s}, Action Index={action_idx}, Action={action}")