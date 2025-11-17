import torch.optim as optim
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import os
import matplotlib.pyplot as plt

hparams = {
    # --- DDPG Agent Params ---
    'lr_actor': 0.0015,
    'lr_critic': 0.00075,
    'gamma': 0.99,            
    'tau': 0.0001,            
    'buffer_size': 8000,
    'batch_size': 128,  
    'reward_threshold': -50,      
    
    # --- Exploration (OU Noise) Params ---
    'mu': 0.0,
    'theta': 0.12,
    'sigma': 0.3,
    'dt': 0.1,
    'decay_rate': 0.995,
    'x0': 0.0,      
}

class DDPG_agent():

    def __init__(self, state_dim, action_dim, action_high, seed, hparams):
        
        # Determine the device (CPU or GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Initializing DDPG Agent on device: {self.device}")

        # Amount of neurons in hidden layer
        hidden_size1 = 256
        hidden_size2 = 128

        # --- Unpack Agent Parameters ---
        self.lr_actor = hparams['lr_actor']
        self.lr_critic = hparams['lr_critic']
        self.gamma = hparams['gamma']
        self.tau = hparams['tau']
        self.buffer_size = hparams['buffer_size']
        self.batch_size = hparams['batch_size']
        self.reward_threshold = hparams['reward_threshold']
        
        # --- Unpack OU Noise Parameters ---
        self.mu = hparams['mu']
        self.theta = hparams['theta']
        self.sigma = hparams['sigma']
        self.dt = hparams['dt']
        self.decay_rate = hparams['decay_rate']
        self.x0 = hparams['x0']

        # Initialize the networks

        # Misc - memory, noise, etc.
        self.memory = replaybuffer(self.buffer_size, state_dim, action_dim, self.batch_size, reward_threshold= self.reward_threshold)
        self.noise = OUNoise(self.mu, self.theta, self.sigma, self.dt, self.x0)
        self.scale_factor = action_high

        ## Initializing Actor networks
        self.mainActor = Actor(state_dim, action_dim, hidden_size1, hidden_size2).to(self.device)
        self.targetActor = Actor(state_dim, action_dim, hidden_size1, hidden_size2).to(self.device)
        
        ## Initializing Critic networks
        self.mainCritic = Critic(state_dim, action_dim, hidden_size1, hidden_size2).to(self.device) 
        self.targetCritic = Critic(state_dim, action_dim, hidden_size1, hidden_size2).to(self.device)

        ## Initializing Optimizers for main Actor/Critic
        self.actor_optimizer = optim.Adam(self.mainActor.parameters(), self.lr_actor)
        self.critic_optimizer = optim.Adam(self.mainCritic.parameters(), self.lr_critic)

        # Copy weights from main -> target (to initialize the same) 
        self.targetActor.load_state_dict(self.mainActor.state_dict())
        self.targetCritic.load_state_dict(self.mainCritic.state_dict()) 

    def act(self, state, add_noise=True):
        
        # Convert NP array to PyTorch Tensor and move to device
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Act in evaluation mode + no gradient
        self.mainActor.eval()
        
        with torch.no_grad():
            # The action is scaled by the network itself, but we pass the high value
            action = self.mainActor.forward(state, scale_factor=self.scale_factor)
        
        # Convert PyTorch Tensor back to NP array on CPU
        action = action.detach().cpu().numpy()

        if add_noise:
            action += self.noise.sample()
            self.noise.sigma = self.noise.sigma * self.decay_rate

        # Clip result to match initial scale factor
        action = np.clip(action, -self.scale_factor, self.scale_factor)

        return action
    
    def learn(self, state, action, rewards, nextStates, dones):

        # Move sampled tensors to device for training
        state = state.to(self.device)
        action = action.to(self.device)
        rewards = rewards.to(self.device)
        nextStates = nextStates.to(self.device)
        dones = dones.to(self.device)

        # Target Q-Value
        with torch.no_grad():
            nextAction = self.targetActor(nextStates, scale_factor=self.scale_factor)         
            q_prime = self.targetCritic(nextStates, nextAction)
            target_y = rewards + self.gamma*q_prime * (1-dones)

        # CRITIC

        # Main Q-Value and loss
        current_q = self.mainCritic(state, action)
        critic_loss = F.mse_loss(current_q, target_y)

        # LEARNING TIME
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ACTOR

        # Q-Values and loss
        actor_actions = self.mainActor(state, scale_factor=self.scale_factor)
        actor_q_values = self.mainCritic(state, actor_actions)
        actor_loss = -1*actor_q_values.mean()

        # LEARNING TIME
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.mainActor, self.targetActor)
        self.soft_update(self.mainCritic, self.targetCritic)

        return critic_loss.item()

    def soft_update(self, main_model, target_model):
        
        # Update targets
        for target_param, main_param in zip(target_model.parameters(), main_model.parameters()):
            
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(main_param.data, alpha=self.tau)

    def save(self, filename):

        checkpoint = {
            'actor_state_dict': self.mainActor.state_dict(),
            'critic_state_dict': self.mainCritic.state_dict(),
        }
        
        torch.save(checkpoint, filename)
    
    def load(self, filename):

        # Load to CPU first, then move to device. This handles both CPU and CUDA loading safely.
        checkpoint = torch.load(filename, map_location=self.device)

        self.mainActor.load_state_dict(checkpoint['actor_state_dict'])
        self.mainCritic.load_state_dict(checkpoint['critic_state_dict'])

class OUNoise:

    def __init__(self, mu, theta, sigma, dt, x0 = 0):

        
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.state = self.x0 * np.ones(1)

    def reset(self):
        
        # Reset state to 0 
        self.state = self.mu * np.ones(self.state.shape)
        
    def sample(self):
        
        # Calculate dW and dx
        dW = np.random.normal(size = self.state.shape)
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma*math.sqrt(self.dt)*dW
        
        self.state += dx

        return self.state
    
class replaybuffer:
    
    def __init__(self, capacity, state_dim, action_dim, batch_size, reward_threshold):
        
        # Buffer size
        self.capacity = capacity
        self.batch_size = batch_size
        self.reward_threshold = reward_threshold
        
        # 5 arrays for SARS
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.nextState = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))

        # Size and index
        self.idx = 0
        self.size = 0

    def store(self, state, action, reward, nextState, done):
        
        # SARS
        self.state[self.idx] = state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.nextState[self.idx] = nextState
        self.dones[self.idx] = done

        # Track size and index
        if self.size<self.capacity:
            self.size += 1

        self.idx += 1
        if self.idx>=self.capacity:
            self.idx = 0

    def sample(self):
        
        # Random idx 
        current_indices = np.arange(self.size)

        high_reward_mask = self.reward[:self.size].squeeze() > self.reward_threshold
        high_reward_indices = current_indices[high_reward_mask]

        if len(high_reward_indices) < self.batch_size:
            # Fallback: If not enough clean data, sample randomly from all data (standard DDPG)
            idx = np.random.randint(0, self.size, self.batch_size)
        else:
            # Main Path: Sample the batch ONLY from the high-reward indices
            idx = np.random.choice(high_reward_indices, self.batch_size, replace=False)

        # Take the SARSD
        states = self.state[idx]
        actions = self.action[idx]
        rewards = self.reward[idx]
        nextStates = self.nextState[idx]
        dones = self.dones[idx]

        # Convert NP array to PyTorch Tensor 
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        nextStates = torch.from_numpy(nextStates).float()
        dones = torch.from_numpy(dones).float()
        
        # NOTE: Tensors are moved to the device (CPU/GPU) in the DDPG_agent.learn() method.
        return states, actions, rewards, nextStates, dones
    
    def __len__(self):
        """Returns the current number of stored experiences."""
        return self.size

class Actor(nn.Module):

    def __init__(self, states, actions, hidden_size1, hidden_size2) -> None:

        super(Actor, self).__init__()

        # Layers
        self.input_size = states
        self.output_size = actions

        self.fc1 = nn.Linear(self.input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, self.output_size)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, scale_factor = 2.0):

        out = self.fc1(x)
        out = self.relu(out)
        out = F.dropout(out, p=0.05, training=self.training)
        out = self.fc2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.05, training=self.training)
        out = self.fc3(out)
        out = self.tanh(out)
        
        scaled_output = out*scale_factor

        return scaled_output

class Critic(nn.Module):
    def __init__(self, states, actions, hidden_size1, hidden_size2) -> None:

        super(Critic, self).__init__()

        # Layers
        self.input_size = states
        self.output_size = actions

        self.fc1 = nn.Linear(self.input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1 + actions, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, self.output_size)

        # Activations
        self.relu = nn.ReLU()

    def forward(self, state, actions):

        out = self.fc1(state)
        out = self.relu(out)
        out = F.dropout(out, p=0.05, training=self.training)
        out = torch.cat([out, actions],dim = 1)
        out = self.fc2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.05, training=self.training)
        out = self.fc3(out)

        return out


# Import Environment

env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  

# Initialize agent and relevant data

hparams = {
    # --- DDPG Agent Params ---
    'lr_actor': 0.0015,
    'lr_critic': 0.00075,
    'gamma': 0.995,           
    'tau': 0.0001,            
    'buffer_size': 8000,
    'batch_size': 256,        
    'reward_threshold': -50,      
    
    # --- Exploration (OU Noise) Params ---
    'mu': 0.0,
    'theta': 0.12,
    'sigma': 0.3,
    'dt': 0.1,
    'decay_rate': 0.995,
    'x0': 0.0,      
}
seed = 42
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high[0]
agent = DDPG_agent(state_dim, action_dim, action_high=action_high, seed=seed, hparams = hparams)

def run(env, episodes = 500, seed: int = 5, agent_instance = None, batch_size: int = 128, target = -500):

    # check for agent
    if agent_instance is None:
        agent_instance = agent

    done = False
    state, _ = env.reset(seed=seed)
    state = np.squeeze(state)

    scores = []

    # Fill replay buffer
    while agent_instance.memory.size < batch_size:
        action = agent_instance.act(state, add_noise=False)
        nextState, reward, terminated, truncated, _ = env.step(action)
        reward = np.squeeze(reward)
        nextState = np.squeeze(nextState)
        done = truncated or terminated
        agent_instance.memory.store(state, action, reward, nextState, done)
        state = nextState
        
        if done:
            state, _ = env.reset(seed=seed)
            state = np.squeeze(state)
            

    # The actual run
    for i in range(episodes):

        total_reward = 0
        state, _ = env.reset()
        state = np.squeeze(state)
        # Reset OU Noise for the start of a new episode
        agent_instance.noise.reset() 

        while True: 

            action = agent_instance.act(state, add_noise=True)
            nextState, reward, terminated, truncated, _ = env.step(action)
            reward = np.squeeze(reward)
            nextState = np.squeeze(nextState)
            total_reward += reward
            done = truncated or terminated
            
            # Store transition and learn
            agent_instance.memory.store(state, action, reward, nextState, done)

            if len(agent_instance.memory) >= batch_size:
                # Sample is done inside the learn method now
                agent_instance.learn(*agent_instance.memory.sample())

            state = nextState


            if done:
                break
        
        scores.append((i+1 ,total_reward))

        if (i + 1) % 10 == 0:
            avg_score_10 = np.mean([s[1] for s in scores[-10:]])
            print(f"Episode: {i+1}/{episodes} | Total Reward: {total_reward:.2f} | Avg Reward (last 10): {avg_score_10:.2f}")

    # Plot and save
    print("Saving plot...")
    x_coords, y_coords = zip(*scores)
    plt.plot(x_coords, y_coords)
    plt.savefig("Latest training plot")
    print("Saved plot!")

    # --- Phase 3: Final Output and Return ---
    print("\nSaving weights...")
    # Ensure 'models' directory exists if running standalone
    os.makedirs('models', exist_ok=True) 
    agent_instance.save(filename="models/weights.pth")
    
    print("Saved!")

    # Calculate average score of the final 100 episodes (or fewer if episodes < 100)
    avg_score_final = np.mean([s[1] for s in scores[-min(episodes, 100):]])
    print(f"Final Average Score (last {min(episodes, 100)} episodes): {avg_score_final:.2f}")
    
    # CRITICAL: Return the final average score for the sweep script
    return avg_score_final

def evaluate_agent(env, agent_instance = None, target = -1000):

    if agent_instance is None:
        agent_instance = agent

    total_reward = 0
    num_episodes = 10


    for i in range(num_episodes):
        state, _ = env.reset()
        state = np.squeeze(state)
        
        episode_reward = 0
        while True: 

            action = agent_instance.act(state, add_noise=False)
            nextState, reward, terminated, truncated, _ = env.step(action)
            reward = np.squeeze(reward)
            nextState = np.squeeze(nextState)
            episode_reward += reward
            done = truncated or terminated
            state = nextState
            
            if done:
                total_reward += episode_reward
                break
                
    avg_reward = total_reward/num_episodes
    if avg_reward > target:
        print(f"for this run, agent scored an average of {avg_reward:.2f}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print("Starting DDPG training run...")
    
    # --- IMPORTANT: Change this number for a full training run! ---
    # Use 500 for a quick test, or 10000 for a deep training run.
    EPISODES = 500 

    # Run the training loop
    final_score = run(env, episodes=EPISODES, agent_instance=agent, batch_size=hparams['batch_size'])
    
    print(f"\nTraining finished after {EPISODES} episodes.")
    print(f"Final {min(EPISODES, 100)}-episode average score: {final_score:.2f}")

    # Optional: Load and evaluate the saved agent for a final confirmation
    print("\nAttempting to load and evaluate the saved model (10 test episodes)...")
    try:
        # Create a new agent instance for testing purposes
        loaded_agent = DDPG_agent(state_dim, action_dim, action_high=action_high, seed=seed, hparams = hparams)
        loaded_agent.load(filename="models/weights.pth")
        evaluate_agent(env, agent_instance=loaded_agent)
    except FileNotFoundError:
        print("Could not find saved weights file to evaluate.")
    
    # Close environment
    env.close()