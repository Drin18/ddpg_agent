import gymnasium as gym
import torch
import numpy as np
import os
import time

# Import the DDPG_agent class, hyperparameters, and the shared evaluation function
from DDPG_Pendulum import DDPG_agent, hparams, evaluate_agent

# --- Configuration ---
env_name = "Pendulum-v1"
weights_path = "models/weights.pth"
num_episodes = 10
seed = 42 

# The run_evaluation function is now removed, and the logic is handled by evaluate_agent 
# imported from DDPG_Pendulum.py.

if __name__ == '__main__':
    
    # 1. Initialize Environment and Parameters
    print(f"Setting up environment: {env_name}")
    
    # Attempt to use 'human' render mode for visualization
    try:
        # Create the human rendering environment
        env = gym.make(env_name, render_mode="human", g=9.81) 
    except Exception as e:
        print(f"Warning: Could not start in 'human' render mode. Falling back to 'rgb_array'. Error: {e}")
        env = gym.make(env_name, render_mode="rgb_array", g=9.81)
    
    # Extract environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # 2. Instantiate the DDPG Agent
    print("Instantiating DDPG Agent for loading...")
    # The agent automatically determines the device (CPU/GPU) upon initialization
    agent = DDPG_agent(state_dim, action_dim, action_high=action_high, seed=seed, hparams=hparams)

    # 3. Load Saved Weights
    if os.path.exists(weights_path):
        print(f"Attempting to load weights from: {weights_path}")
        try:
            # The agent's load method handles device mapping automatically
            agent.load(filename=weights_path)
            print("Weights loaded successfully!")
            
            # 4. Run Evaluation using the shared function
            print(f"\n--- Starting Evaluation: {num_episodes} episodes ---")
            
            # Temporarily modify the evaluation loop to include time.sleep for visualization.
            # NOTE: This overrides the default environment step logic, so we must manually 
            # step through the environment here instead of calling the function directly.
            
            total_scores = []
            
            for i in range(1, num_episodes + 1):
                # Reset the environment for a new episode
                state, _ = env.reset(seed=seed + i) 
                state = np.squeeze(state)
                episode_reward = 0
                step = 0

                while True: 
                    # 1. Use the agent to select an action deterministically (add_noise=False)
                    action = agent.act(state, add_noise=False)
                    
                    # 2. Step the environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    reward = np.squeeze(reward)
                    next_state = np.squeeze(next_state)
                    
                    # 3. Update state and rewards
                    episode_reward += reward
                    done = terminated or truncated
                    state = next_state
                    step += 1
                    
                    # CRITICAL: Add the pause for visualization
                    time.sleep(0.01) 
                    
                    if done:
                        break
                
                total_scores.append(episode_reward)
                print(f"Episode {i}/{num_episodes}: Score = {episode_reward:.2f} (Steps: {step})")

            # Final summary
            avg_score = np.mean(total_scores)
            print(f"\nEvaluation Complete. Average Score over {num_episodes} episodes: {avg_score:.2f}")

            
        except Exception as e:
            print(f"Error loading weights or running evaluation: {e}")
            print("Please ensure your 'models/weights.pth' file exists and is not corrupted, and that DDPG_Pendulum.py is correct.")
    else:
        print(f"Error: Weights file not found at '{weights_path}'.")
        print("Please run your training script (DDPG_Pendulum.py) first to generate the weights.")
    
    # 5. Close the environment
    env.close()