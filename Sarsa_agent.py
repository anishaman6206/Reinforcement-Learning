# Sarsa Agent Implementation for CliffWalking-v0 Environment
import gym  
import numpy as np  
import pickle as pkl  

# --- Environment Setup ---
# Creates the "CliffWalking-v0" environment from OpenAI Gym.
cliffEnv = gym.make("CliffWalking-v0")

# --- Q-Table Initialization ---
# The Q-table stores the Q-values for each state-action pair.
# Shape: (48 states, 4 actions). Initialized to all zeros.
q_table = np.zeros(shape=(48, 4))

# --- Hyperparameters ---
EPSILON = 0.1  # Exploration rate (10% chance of random action).
ALPHA = 0.1    # Learning rate.
GAMMA = 0.9    # Discount factor.
NUM_EPISODES = 500  # Total number of training episodes.

# --- Epsilon-Greedy Policy ---
# This function decides which action to take from a given state.
def policy(state, explore=0.0):
    """
    Args:
        state: The current state of the agent.
        explore: The probability of taking a random action (epsilon).

    Returns:
        The action to take.
    """
    # Exploit: Choose the action with the highest Q-value.
    action = int(np.argmax(q_table[state]))
    
    # Explore: With probability 'explore', choose a random action.
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
        
    return action

# --- Training Loop ---
# The main loop where the agent learns.
for episode in range(NUM_EPISODES):

    # --- Episode Initialization ---
    done = False
    total_reward = 0
    episode_length = 0

    # Reset the environment to the starting state.
    state = cliffEnv.reset()

    # **CRITICAL SARSA STEP 1**: Choose the first action *before* the loop starts.
    # The action is chosen using the epsilon-greedy policy.
    action = policy(state, EPSILON)

    # --- Step Loop ---
    # This loop runs for each step within a single episode.
    while not done:
        # 1. Take Action: Perform the action chosen before the loop (or in the previous iteration).
        next_state, reward, done, _ = cliffEnv.step(action)

        # 2. **CRITICAL SARSA STEP 2**: Choose the *next* action from the *next* state, again using the policy.
        # This is the 'A' in S-A-R-S-A that will be used in the next iteration.
        next_action = policy(next_state, EPSILON)

        # 3. SARSA Update Rule: This is the core of the algorithm.
        # It updates the Q-value for the (state, action) pair.
        # The update is based on the reward and the Q-value of the (next_state, next_action) pair.
        # This is what makes SARSA "on-policy" - it learns based on the actions it is actually taking.
        current_q = q_table[state][action]
        next_q = q_table[next_state][next_action]
        new_q = current_q + ALPHA * (reward + GAMMA * next_q - current_q)
        q_table[state][action] = new_q

        # 4. Update State and Action: The next state and next action become the current ones for the next iteration.
        state = next_state
        action = next_action

        # Accumulate reward and step count for logging.
        total_reward += reward
        episode_length += 1
        
    # Print the results of the episode.
    print(f"Episode: {episode}, Length: {episode_length}, Total Reward: {total_reward}")

# --- Cleanup and Save ---
# Close the environment.
cliffEnv.close()

# Save the trained Q-table to a file.
pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("\nTraining Complete. Q-table saved to 'sarsa_q_table.pkl'")
