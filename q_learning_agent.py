# Q-Learning Implementation for CliffWalking Environment

import gym  # The main OpenAI Gym library for reinforcement learning environments
import numpy as np  
import pickle as pkl  

# --- Environment Setup ---
# Creates the "CliffWalking-v0" environment from OpenAI Gym.
# This is a 4x12 grid world where the agent must go from a start state 'S' to a goal state 'G'
# without falling off a cliff.
cliffEnv = gym.make("CliffWalking-v0")

# --- Q-Table Initialization ---
# The Q-table is a data structure that stores the expected future rewards (Q-values) for each state-action pair.
# The shape is (number of states, number of actions).
# For CliffWalking, there are 48 states (4x12 grid) and 4 possible actions (Up, Down, Left, Right).
# We initialize all Q-values to zero to start.
q_table = np.zeros(shape=(48, 4))

# --- Hyperparameters ---
# These parameters control the learning process.
EPSILON = 0.1  # Exploration rate: The probability of choosing a random action instead of the best one. 10% chance.
ALPHA = 0.1    # Learning rate: How much we update our Q-values based on new information.
GAMMA = 0.9    # Discount factor: How much we value future rewards. A value closer to 1 means more focus on long-term rewards.
NUM_EPISODES = 500  # The total number of episodes (games) to play during training.

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
    # Exploit: Choose the action with the highest Q-value for the current state.
    action = int(np.argmax(q_table[state]))
    
    # Explore: With a probability of 'explore' (epsilon), choose a random action instead.
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
        
    return action


# --- Training Loop ---
# The main loop where the agent learns. It runs for a total of NUM_EPISODES.
for episode in range(NUM_EPISODES):

    # --- Episode Initialization ---
    done = False  # A flag to check if the episode has ended (e.g., reached the goal or fallen off the cliff).
    total_reward = 0  # The sum of rewards collected in this episode.
    episode_length = 0  # The number of steps taken in this episode.

    # Reset the environment to get the starting state for the new episode.
    state = cliffEnv.reset()

    # --- Step Loop ---
    # This loop runs for each step within a single episode.
    while not done:
        # 1. Choose Action: Select an action based on the current state and the epsilon-greedy policy.
        action = policy(state, EPSILON)

        # 2. Take Action: Perform the chosen action in the environment.
        # It returns the next state, the reward received, whether the episode is done, and other info.
        next_state, reward, done, _ = cliffEnv.step(action)

        # 3. Q-Learning Update Rule: This is the core of the algorithm.
        # It updates the Q-value for the (state, action) pair.
        # The update is based on the immediate reward and the *maximum possible Q-value* from the next state.
        # This is what makes Q-Learning "off-policy" - it learns about the optimal path, not the path it's taking.
        current_q = q_table[state][action]
        max_future_q = np.max(q_table[next_state])
        new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
        q_table[state][action] = new_q

        # 4. Update State: The next state becomes the current state for the next step.
        state = next_state

        # Accumulate reward and step count for logging.
        total_reward += reward
        episode_length += 1
        
    # Print the results of the episode.
    print(f"Episode: {episode}, Length: {episode_length}, Total Reward: {total_reward}")

# --- Cleanup and Save ---
# Close the environment to free up resources.
cliffEnv.close()

# Save the trained Q-table to a file using pickle.
# This allows us to load and use the learned policy later without retraining.
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print("\nTraining Complete. Q-table saved to 'q_learning_q_table.pkl'")
