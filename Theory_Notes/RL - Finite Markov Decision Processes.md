# Reinforcement Learning: Finite Markov Decision Processes
## Complete Study Notes

---

## The Big Leap - From Bandits to Sequential Decisions

### **The Evolution: Bandits → Full RL**

| **Multi-Armed Bandits** | **Markov Decision Processes** |
|-------------------------|--------------------------------|
| Action → Reward → Reset | Action → Reward + New State |
| Single isolated decisions | Sequential decisions with consequences |
| No future impact | Actions change future situations |
| Non-associative | Associative (context matters) |

### **The Robot Maze Analogy**

**Scenario:** Robot in maze with cheese (+1), shocks (-1), empty squares (0)

- **Goal:** Not just next cheese, but optimal long-term path
- **Challenge:** Every move changes position and future options

---

## The Agent-Environment Interface

### **The Core Loop**

```
Time Steps: t = 0, 1, 2, 3, ...

At each step t:
Agent observes State S_t → Chooses Action A_t
Environment provides Reward R_{t+1} + New State S_{t+1}
```

### **Key Components**

**State ($S_t$)**
- Complete description of agent's situation
- Contains all relevant information for decisions
- Example: (x,y) coordinates in maze

**Action ($A_t$)**  
- Choice from available action set
- Example: {North, South, East, West}

**Reward ($R_{t+1}$)**
- Numerical feedback for action quality
- Subscript t+1 indicates it results from action at time t

**Trajectory**
- Sequence of experience: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, ...$

---

## The Markov Property - The "Memoryless" Rule

### **Core Principle**

> **"The future is independent of the past, given the present"**

**Mathematical Expression:**

$$P[S_{t+1} = s', R_{t+1} = r | S_t, A_t] = P[S_{t+1} = s', R_{t+1} = r | S_0, A_0, ..., S_t, A_t]$$

### **Practical Meaning**

- **Current state contains ALL relevant history**
- **No need to remember entire path**
- **Decision depends only on where you are NOW**

**Maze Example:** Robot only needs current (x,y) position, not the path taken to get there

### **Why This Matters**

- Massive simplification of decision-making
- Enables tractable solutions
- Foundation of all MDP algorithms

---

## Part 2: Defining Success - The Return Function

### **Two Types of Tasks**

**1. Episodic Tasks**
- **Natural ending point**
- Examples: Chess game, maze run
- **Simple sum return:**

$$G_t = R_{t+1} + R_{t+2} + ... + R_T$$

where T = terminal time step

**2. Continuing Tasks**
- **No natural end**
- Examples: Robot balancing, power grid management
- **Problem:** Infinite rewards → unusable goal

### **The Solution: Discounting**

**Core Idea:** *Earlier rewards are more valuable than later ones*

**Discount Factor γ (gamma)**
- Value between 0 and 1
- Controls "farsightedness" of agent

**Discounted Return Formula:**

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### **Discount Factor Interpretation**

| γ Value | Agent Behavior | Example |
|---------|----------------|---------|
| γ = 0 | Myopic (immediate only) | Only cares about $R_{t+1}$ |
| γ = 0.5 | Moderate foresight | Future reward worth 50% less per step |
| γ = 0.9 | High foresight | Future reward worth 90% per step |
| γ → 1 | Maximum foresight | All future rewards nearly equal |

**Mathematical Guarantee:** If γ < 1 and rewards bounded → $G_t$ is finite

---

## Part 3: The Agent's Brain - Policies and Value Functions

### **Policy (π) - The Agent's Strategy**

**Definition:** Mapping from states to action probabilities

$$\pi(a|s) = P[A_t = a | S_t = s]$$

**Types:**
- **Deterministic:** π(s) = specific action
- **Stochastic:** π(a|s) = probability distribution over actions

### **Value Functions - Measuring Strategy Quality**

**1. State-Value Function $v_\pi(s)$**

*"How good is it to be in this state?"*

$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

- Expected return starting from state s
- Following policy π thereafter

**2. Action-Value Function $q_\pi(s,a)$**

*"How good is it to take this action from this state?"*

$$q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

- Expected return after taking action a in state s
- Then following policy π

**Relationship Between Value Functions:**

$$v_\pi(s) = \sum_a \pi(a|s) \cdot q_\pi(s,a)$$

---

## Part 4: The Bellman Equation - The Secret Sauce

### **The Core Insight**

> **"Value of current state = Average immediate reward + Average discounted value of next states"**

### **Step-by-Step Example: 2×2 Grid World**

**Setup:**
```
+-------+-------+
|   A   |   B   |
+-------+-------+
| C(Pit)| D(Goal)|
+-------+-------+
```

**Rules:**
- States: {A, B, C, D}
- Actions: {North, South, East, West}
- Rewards: Goal D = +10, Pit C = -10, others = -1
- Terminal states: v(C) = v(D) = 0
- Discount: γ = 0.9
- Policy: Random (25% each direction)

### **Building the Bellman Equation for State A**

**Move Analysis:**

1. **North from A:** Hit wall → stay in A
   - Value = -1 + 0.9 × v(A)

2. **West from A:** Hit wall → stay in A  
   - Value = -1 + 0.9 × v(A)

3. **East from A:** Move to B
   - Value = -1 + 0.9 × v(B)

4. **South from A:** Fall in pit C
   - Value = -10 + 0.9 × 0 = -10

**Complete Bellman Equation:**

$$v(A) = 0.25 \times [-1 + 0.9 \times v(A)] + 0.25 \times [-1 + 0.9 \times v(A)] + 0.25 \times [-1 + 0.9 \times v(B)] + 0.25 \times [-10]$$

### **System of Equations**

After simplification:

1. $v(A) = 0.45 \times v(A) + 0.225 \times v(B) - 3.25$
2. $v(B) = 0.225 \times v(A) + 0.45 \times v(B) + 1.75$

**Solution:** v(A) ≈ -5.57, v(B) ≈ -2.14

---

## Part 5: Dynamic Programming - Iterative Solutions

### **Requirements**

- **Perfect model** of environment
- **All transition probabilities** known
- **Finite state space**

### **Core Process: Policy Evaluation**

**Iterative Method:**

1. Start with arbitrary values: $v_0(s) = 0$ for all s

2. Apply Bellman equation iteratively:

$$v_{k+1}(s) = \sum_a \pi(a|s) \times \left[\sum_{s',r} p(s',r|s,a) \times (r + \gamma \times v_k(s'))\right]$$

**Breaking down the equation:**

- **$\pi(a|s)$:** Probability of taking action a in state s under policy π
- **$\sum_a$:** Sum over all possible actions from state s
- **$p(s',r|s,a)$:** Transition probability to state s' with reward r, given action a from state s
- **$\sum_{s',r}$:** Sum over all possible next states and rewards

**Interpretation:** The outer sum weights each action by its probability under the policy, then computes the expected value of that action.

### **Complete Worked Example: 2×2 Grid**

**MDP Setup (Detailed):**

- **States:** A(0), B(1), C(2,terminal), D(3,terminal)
- **Actions:** {North, West, East, South} - deterministic transitions
- **Specific Rewards:**
  - Move into Goal D: +10
  - Move into Pit C: -10
  - All other moves (including walls): -1
- **Terminal Values:** v(C) = v(D) = 0

**Deterministic Transitions:**

```
From A: N/W → stay A (-1), E → B (-1), S → C (-10)
From B: N/E → stay B (-1), W → A (-1), S → D (+10)
```

---

## **Method 1: Policy Iteration - Complete Walkthrough**

### **Step 1: Policy Evaluation (Solve Linear System)**

**Random Policy π₀:** Each action probability = 0.25

**Expected Immediate Rewards ($r_\pi$):**

**What is $r_\pi(s)$?** The expected immediate reward when starting in state s and following policy π

**For State A:**

$$r_\pi(A) = \sum_a \pi(a|A) \times [\text{expected immediate reward for action a from A}]$$

$$r_\pi(A) = 0.25 \times (-1) + 0.25 \times (-1) + 0.25 \times (-1) + 0.25 \times (-10) = -3.25$$

**For State B:**

$$r_\pi(B) = 0.25 \times (-1) + 0.25 \times (-1) + 0.25 \times (-1) + 0.25 \times (+10) = +1.75$$

**Mathematical Definition:**

$$r_\pi(s) = \sum_a \pi(a|s) \times \sum_{s',r} p(s',r|s,a) \times r$$

This is the **expected immediate reward vector** used in the linear system approach to policy evaluation.

**Transition Probabilities Under π₀:**

```
From A: 50% stay A, 25% go to B, 25% terminal
From B: 50% stay B, 25% go to A, 25% terminal

P_π = [[0.5,  0.25],    (A→A, A→B)
       [0.25, 0.5 ]]     (B→A, B→B)
```

**We are solving the Bellman expectation equation in matrix form:**

$$v_\pi = r_\pi + \gamma P_\pi v_\pi$$

**Where:**
- **$v_\pi$** = column vector of state values (size: number of states × 1)
- **$r_\pi$** = column vector of expected **one-step rewards** under policy π (same size)
- **$P_\pi$** = transition probability matrix under policy π (size: number of states × number of states)

**Step 1: Rearrange**

$$v_\pi - \gamma P_\pi v_\pi = r_\pi$$

**Factor $v_\pi$:**

$$(I - \gamma P_\pi) v_\pi = r_\pi$$

**Bellman Linear System:** $(I - \gamma P_\pi)v = r_\pi$

$$I - 0.9P_\pi = \begin{bmatrix} 0.55 & -0.225 \\ -0.225 & 0.55 \end{bmatrix}$$

Right-hand side: $[-3.25, 1.75]$

**Complete Linear System:**

$$(I - \gamma P_\pi) \times v_\pi = r_\pi$$

$$\begin{bmatrix} 0.55 & -0.225 \\ -0.225 & 0.55 \end{bmatrix} \times \begin{bmatrix} v(A) \\ v(B) \end{bmatrix} = \begin{bmatrix} -3.25 \\ 1.75 \end{bmatrix}$$

**This expands to a system of 2 linear equations:**

1. $0.55v(A) - 0.225v(B) = -3.25$
2. $-0.225v(A) + 0.55v(B) = 1.75$

**Solving 2×2 System:**

$$\text{Determinant} = 0.55^2 - 0.225^2 = 0.3025 - 0.050625 = 0.251875$$

$$v(A) = \frac{0.55 \times (-3.25) - (-0.225) \times 1.75}{0.251875} = \frac{-1.7875 + 0.39375}{0.251875} = -5.53$$

$$v(B) = \frac{(-0.225) \times (-3.25) + 0.55 \times 1.75}{0.251875} = \frac{-0.73125 + 0.9625}{0.251875} = 0.92$$

**Solution:** v(A) ≈ -5.53, v(B) ≈ 0.92

### **Step 2: Policy Improvement**

**Calculate Q-values using $v_{\pi_0}$:**

**From State A:**

- $q(A,N) = -1 + 0.9 \times (-5.53) = -5.98$
- $q(A,W) = -1 + 0.9 \times (-5.53) = -5.98$
- $q(A,E) = -1 + 0.9 \times (0.92) = -0.17$ ← **Best!**
- $q(A,S) = -10 + 0.9 \times 0 = -10.00$

**From State B:**

- $q(B,N) = -1 + 0.9 \times (0.92) = -0.17$
- $q(B,E) = -1 + 0.9 \times (0.92) = -0.17$
- $q(B,W) = -1 + 0.9 \times (-5.53) = -5.98$
- $q(B,S) = +10 + 0.9 \times 0 = +10.00$ ← **Best!**

**New Policy π₁:** East from A, South from B

### **Step 3: Evaluate New Policy π₁**

**Under deterministic policy (A→E, B→S):**

- $v(B) = +10 + 0.9 \times 0 = 10$ (B→D directly, terminal)
- $v(A) = -1 + 0.9 \times 10 = 8$ (A→B, then B→D optimally)

**Policy Improvement Check:** 
- A: Still chooses East (8.0 > all others)
- B: Still chooses South (10.0 > all others)

→ **Policy Stable!**

---

## **Method 2: Value Iteration - Complete Walkthrough**

**Update Rule:** 

v_{k+1}(s) = max_a [r(s,a) + γv_k(next_state)]

### **Iteration Sequence:**

**k=0:** $v_0(A) = 0$, $v_0(B) = 0$

**k=1 (Iteration 0→1):**

**State B:**

- $q(B,N) = -1 + 0.9 \times v_0(B) = -1 + 0.9 \times 0 = -1.0$ (North hits wall, stay B)
- $q(B,E) = -1 + 0.9 \times v_0(B) = -1 + 0.9 \times 0 = -1.0$ (East hits wall, stay B)
- $q(B,W) = -1 + 0.9 \times v_0(A) = -1 + 0.9 \times 0 = -1.0$ (West goes to A)
- $q(B,S) = +10 + 0.9 \times v_0(D) = +10 + 0.9 \times 0 = 10.0$ (South goes to Goal D)

$$v_1(B) = \max\{-1.0, -1.0, -1.0, 10.0\} = 10.0$$

**State A:**

- $q(A,N) = -1 + 0.9 \times v_0(A) = -1 + 0.9 \times 0 = -1.0$ (North hits wall, stay A)
- $q(A,W) = -1 + 0.9 \times v_0(A) = -1 + 0.9 \times 0 = -1.0$ (West hits wall, stay A)
- $q(A,E) = -1 + 0.9 \times v_0(B) = -1 + 0.9 \times 0 = -1.0$ (East goes to B)
- $q(A,S) = -10 + 0.9 \times v_0(C) = -10 + 0.9 \times 0 = -10.0$ (South goes to Pit C)

$$v_1(A) = \max\{-1.0, -1.0, -1.0, -10.0\} = -1.0$$

**After k=1:** $v_1 = [A: -1.0, B: 10.0]$

**k=2 (Iteration 1→2):**

**State B:**

- $q(B,N) = -1 + 0.9 \times v_1(B) = -1 + 0.9 \times 10.0 = 8.0$ (Stay B, future value 10.0)
- $q(B,E) = -1 + 0.9 \times v_1(B) = -1 + 0.9 \times 10.0 = 8.0$ (Stay B, future value 10.0)
- $q(B,W) = -1 + 0.9 \times v_1(A) = -1 + 0.9 \times (-1.0) = -1.9$ (Go to A, future value -1.0)
- $q(B,S) = +10 + 0.9 \times 0 = 10.0$ (Direct to Goal, immediate +10)

$$v_2(B) = \max\{8.0, 8.0, -1.9, 10.0\} = 10.0$$

**State A:**

- $q(A,N) = -1 + 0.9 \times v_1(A) = -1 + 0.9 \times (-1.0) = -1.9$ (Stay A, future value -1.0)
- $q(A,W) = -1 + 0.9 \times v_1(A) = -1 + 0.9 \times (-1.0) = -1.9$ (Stay A, future value -1.0)
- $q(A,E) = -1 + 0.9 \times v_1(B) = -1 + 0.9 \times 10.0 = 8.0$ (Go to B, future value 10.0)
- $q(A,S) = -10 + 0.9 \times 0 = -10.0$ (Go to Pit, immediate -10)

$$v_2(A) = \max\{-1.9, -1.9, 8.0, -10.0\} = 8.0$$

**After k=2:** $v_2 = [A: 8.0, B: 10.0]$

**k=3 (Convergence Check):**

**State B:**

- $q(B,S) = +10 + 0.9 \times 0 = 10.0$ (Still best)
- $q(B,N) = -1 + 0.9 \times 8.0 = 8.0$ (Other actions still worse)

$v_3(B) = 10.0$ (unchanged)

**State A:**

- $q(A,E) = -1 + 0.9 \times 10.0 = 8.0$ (Still best)

$v_3(A) = 8.0$ (unchanged)

**Converged in 3 iterations!**

### **Extract Optimal Policy:**

**From final Q-values:**
- **State A:** East gives 8.0 (best) → π*(A) = East
- **State B:** South gives 10.0 (best) → π*(B) = South

**Key Insight:** At B, South (direct to goal = 10.0) beats all other actions (8.0)

---

## **Results Comparison**

| Method | Iterations | Final Values | Optimal Policy |
|--------|------------|--------------|----------------|
| **Policy Iteration** | 2 policies | v*(A)=8, v*(B)=10 | π*(A)=East, π*(B)=South |
| **Value Iteration** | 3 iterations | v*(A)=8, v*(B)=10 | π*(A)=East, π*(B)=South |

**Final Optimal Strategy Visualization:**

```
+-------+-------+
|  A →  |  B ↓  |  ← A goes East, B goes South
+-------+-------+
|  C(-) |  D(+) |  
+-------+-------+
v*(A)=8, v*(B)=10, v*(C)=v*(D)=0
```

### **Key Insights**

**Why B→South (not East)?**

- **South:** Direct to goal = +10 + 0.9×0 = **10.0**
- **East:** Hit wall, stay B = -1 + 0.9×10 = **8.0**
- **South > East** at state B

**Optimal Path:** A→East→B, then B→South→D (total return from A = 8)

**Policy Iteration:**
- Fewer policy updates (2 vs 3 iterations)
- Each evaluation step computationally expensive (linear system)
- Better when evaluation cost is manageable

**Value Iteration:**
- More iterations but simpler updates
- Combines evaluation + improvement in each step
- Better for large state spaces or when full policy evaluation is costly

**Both Methods:**
- Guaranteed convergence to same optimal solution
- Demonstrate GPI principle in action
- Show how value functions guide policy improvement

---

## Part 6: Main DP Algorithms

### **Algorithm 1: Policy Iteration**

**Process:**

$$\pi_0 \rightarrow [\text{Evaluate}] \rightarrow v_{\pi_0} \rightarrow [\text{Improve}] \rightarrow \pi_1 \rightarrow [\text{Evaluate}] \rightarrow v_{\pi_1} \rightarrow ...$$

**Steps:**
1. **Initialize:** Arbitrary policy π
2. **Policy Evaluation:** Fully compute $v_\pi$ (inner loop until convergence)
3. **Policy Improvement:** Make π greedy w.r.t. $v_\pi$
4. **Repeat** until policy stable

**Characteristics:**
- **Thorough but slow**
- **Explicit policy maintained**
- **Guaranteed convergence to optimal**

### **Algorithm 2: Value Iteration**

**Key Insight:** Combine evaluation and improvement in single update

**Update Rule:**

$$v_{k+1}(s) = \max_a \left\{\sum_{s',r} p(s',r|s,a) \times [r + \gamma \times v_k(s')]\right\}$$

**Process:**
1. **Initialize:** Arbitrary value function
2. **Value Updates:** Apply rule to all states
3. **Repeat** until convergence
4. **Extract Policy:** One final greedy improvement

**Characteristics:**
- **Faster than Policy Iteration**
- **No explicit policy during updates**
- **Direct optimization of value function**

---

## Bellman Equations Summary

| **Algorithm / Purpose** | **Bellman Equation (State Update Form)** | **Matrix Form** | **Notes / Usage** |
|------------------------|-------------------------------------------|-----------------|-------------------|
| **Policy Evaluation** | $v_{k+1}(s) = \sum_a \pi(a\|s) [r(s,a) + \gamma\sum_{s'} P(s'\|s,a) v_k(s')]$ | $v_\pi = r_\pi + \gamma P_\pi v_\pi$ | Compute value of a **given policy**. Iterative updates or exact solution via matrix inversion. |
| **Policy Iteration** | Same as Policy Evaluation: $v_{k+1}(s) = \sum_a \pi(a\|s) [r(s,a) + \gamma\sum_{s'} P(s'\|s,a) v_k(s')]$ | $v_\pi = r_\pi + \gamma P_\pi v_\pi$ | Alternate between evaluating $v_\pi$ and improving policy π. |
| **Value Iteration** | $v_{k+1}(s) = \max_a [r(s,a) + \gamma\sum_{s'} P(s'\|s,a) v_k(s')]$ | $v_{k+1} = \max_a (r_a + \gamma P_a v_k)$ | Update **values directly** toward optimal $v^*$; greedy policy extracted after convergence. |
| **Bellman Optimality** | $v^*(s) = \max_a [r(s,a) + \gamma\sum_{s'} P(s'\|s,a) v^*(s')]$ | $v^* = \max_a (r_a + \gamma P_a v^*)$ | Defines **optimal value function**, basis of Value Iteration and Q-learning. |

### **Key Distinctions:**

- **Policy Evaluation/Iteration:** Uses **weighted sum** $\left(\sum_a \pi(a|s)\right)$ - considers all actions according to policy probabilities
- **Value Iteration/Optimality:** Uses **maximum** $(\max_a)$ - always picks the best action
- **Matrix forms** show how individual state updates can be vectorized for computational efficiency

---

## Part 7: Limitations and Real-World Challenges

### **Catch 1: Perfect Model Requirement**

**Problem:** Need complete knowledge of environment
- **Transition probabilities:** $p(s',r|s,a)$
- **Reward structure:** Known in advance

**Real-World Reality:** Most environments unknown
- Robot learning to walk
- Stock trading algorithms
- Game playing without rulebook

### **Catch 2: Curse of Dimensionality**

**Problem:** Exponential growth in state space

**Examples:**
- **Chess:** $>10^{47}$ possible states
- **Go:** $>10^{170}$ possible states
- **Real robotics:** Continuous state spaces

**Computational Challenge:** Even at 1M states/second → longer than universe age

---

## The Big Picture: Generalized Policy Iteration (GPI)

### **Conceptual Framework**

**Two Competing/Cooperating Forces:**

**1. Policy Evaluation Force**
- Makes value function consistent with current policy
- "Pulls" V toward true $v_\pi$

**2. Policy Improvement Force**  
- Makes policy greedy with respect to current values
- "Pulls" π toward best strategy for V

### **The GPI Process**

$$\text{Current }(V,\pi) \rightarrow \text{Evaluation} \rightarrow \text{Policy Improvement} \rightarrow \text{Better }(V,\pi) \rightarrow ...$$

**Key Insight:** Forces seem to compete but actually cooperate
- **Short term:** Make each other "wrong"
- **Long term:** Converge to optimal solution

### **Algorithm Variations**

- **Policy Iteration:** Complete evaluation steps
- **Value Iteration:** Interleaved small steps  
- **Advanced methods:** Even finer-grained steps

---

## Summary and Bridge to Future

### **What We've Learned**

**MDP Framework:**
- **Sequential decision problems** with state consequences
- **Markov Property** enables tractable solutions
- **Return maximization** as fundamental goal

**Mathematical Tools:**
- **Bellman equations** link current and future values
- **Policy and value functions** measure strategy quality
- **Dynamic Programming** provides optimal solutions

**Key Algorithms:**
- **Policy Iteration:** Thorough evaluation → improvement cycle
- **Value Iteration:** Combined evaluation/improvement updates
- **GPI:** General framework underlying all RL methods

### **Critical Limitations**

1. **Model requirement:** Perfect environment knowledge needed
2. **Scalability:** Exponential state space growth
3. **Practicality:** Most real problems don't fit assumptions

### **What's Next: Model-Free Methods**

**The Big Question:** *What when we don't have the map?*

**Answer:** Model-free reinforcement learning
- **Learn through trial and error**
- **No environment model required**  
- **Foundation of modern AI breakthroughs**

**Coming Up:**
- **Temporal Difference Learning**
- **Q-Learning**
- **Policy Gradient Methods**
- **Deep Reinforcement Learning**

---

## Key Formulas Reference

**Discounted Return:**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**State-Value Function:**
$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

**Action-Value Function:**
$$q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**Bellman Equation for $v_\pi$:**
$$v_\pi(s) = \sum_a \pi(a|s) \times \sum_{s',r} p(s',r|s,a) \times [r + \gamma \times v_\pi(s')]$$

**Value Iteration Update:**
$$v_{k+1}(s) = \max_a \left\{\sum_{s',r} p(s',r|s,a) \times [r + \gamma \times v_k(s')]\right\}$$

---

## Study Tips

1. **Master the maze example** - concrete foundation for abstract concepts
2. **Understand Bellman intuition** - value = immediate + discounted future
3. **Trace through algorithms** - follow Policy/Value Iteration step-by-step
4. **Recognize limitations** - motivates need for model-free methods
5. **See GPI everywhere** - underlying pattern
