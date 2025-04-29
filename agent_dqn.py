import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#dqn neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#agent class for training
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        #initialize policy network and target network
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.update_target_network()

        #optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        #experience replay memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 64

        #discount factor
        self.gamma = 0.99

        #epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        #device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def update_target_network(self):
        #copy policy network parameters to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        #epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        #store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        #skip training if not enough samples
        if len(self.memory) < self.batch_size:
            return

        #sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        #current q values
        curr_q = self.policy_net(states).gather(1, actions)

        #target q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + (1 - dones) * self.gamma * next_q

        #compute loss
        loss = self.loss_fn(curr_q, expected_q)

        #optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
