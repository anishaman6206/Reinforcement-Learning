# Reinforcement Learning: Temporal-Difference (TD) Learning
## Complete Study Notes

## The Impatient Learner - Beyond Monte Carlo

### **The Evolution: Monte Carlo → Temporal-Difference**

| **Monte Carlo Methods** | **Temporal-Difference Methods** |
|-------------------------|--------------------------------|
| Wait for episode completion | Learn from every step |
| Patient learner | Impatient learner |
| Uses actual returns $G_t$ | Uses estimated returns $R_{t+1} + \gamma V(S_{t+1})$ |
| High variance, unbiased | Low variance, potentially biased |
| Episode-by-episode learning | Step-by-step learning |

### **The Chess Learning Analogy**
**Monte Carlo Approach:**
- Make brilliant move on turn 5
- Make blunder on turn 40 → lose game
- **Learning:** "Loss" → marks ALL moves (including brilliant one) as bad
- **Problem:** Cannot distinguish good moves from bad moves within episode

**TD Approach:**
- Make brilliant move on turn 5 → immediately see improved position
- **Learning:** Update value of previous position immediately based on new position
- **Advantage:** Can give credit/blame to individual moves as they happen

### **The Core Insight**
> **"Learn from the journey, not just the destination"**

TD learning combines the best of both worlds:
- **Model-free** like Monte Carlo (no environment model needed)
- **Bootstrapping** like Dynamic Programming (learns from own estimates)

## The Core of TD - Bootstrapping

### **The Fundamental Concept**
**Bootstrapping:** Update our guess using other guesses

**Monte Carlo Update:**
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
- **Target:** $G_t$ (actual return - must wait for episode end)

**TD(0) Update:**
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### **Breaking Down the TD Update**

**TD Target:** $R_{t+1} + \gamma V(S_{t+1})$
- **$R_{t+1}$:** Immediate reward (real, concrete information)
- **$\gamma V(S_{t+1})$:** Discounted value of next state (bootstrapped estimate)

**TD Error:** $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
- Difference between new estimate (TD target) and old estimate
- Learning signal that drives the update

### **Step-by-Step Example: Random Walk**

**Environment Setup:**
```
[Term 0] ←→ A ←→ B ←→ C ←→ [Term +1]
                  ↑
                Start
```

**Rules:**
- Start in state B, move left/right with 50% probability each
- **Rewards:** Left terminal = 0, Right terminal = +1, all steps = 0
- **Discount:** $\gamma = 1$, **Learning rate:** $\alpha = 0.1$
- **True values:** $v(A) = 0.25$, $v(B) = 0.5$, $v(C) = 0.75$

**Initial Knowledge:** $V(A) = V(B) = V(C) = 0.5$

### **Episode 1: B → A → Term(0)**

**Step 1: B → A**
$$\delta = [0 + 1 \times V(A)] - V(B) = [0 + 0.5] - 0.5 = 0$$
No update (values equal, no surprise)

**Step 2: A → Term(0)**
$$\delta = [0 + 1 \times 0] - V(A) = 0 - 0.5 = -0.5$$
$$V(A) \leftarrow 0.5 + 0.1 \times (-0.5) = 0.45$$

**After Episode 1:** $V(A) = 0.45$, $V(B) = 0.5$, $V(C) = 0.5$

### **Episode 2: B → C → Term(+1)**

**Step 1: B → C**
$$\delta = [0 + 1 \times V(C)] - V(B) = [0 + 0.5] - 0.5 = 0$$
No update

**Step 2: C → Term(+1)**
$$\delta = [1 + 1 \times 0] - V(C) = 1 - 0.5 = 0.5$$
$$V(C) \leftarrow 0.5 + 0.1 \times 0.5 = 0.55$$

**After Episode 2:** $V(A) = 0.45$, $V(B) = 0.5$, $V(C) = 0.55$

### **Episode 3: B → C → B → A → Term(0)**

**Step 1: B → C**
$$\delta = [0 + 1 \times V(C)] - V(B) = [0 + 0.55] - 0.5 = 0.05$$
$$V(B) \leftarrow 0.5 + 0.1 \times 0.05 = 0.505$$

**Key Insight:** Agent updated $V(B)$ mid-episode using improved estimate of $V(C)$!

**Continuing through remaining steps:**
- Information "trickles back" through state space
- Each step provides learning opportunity
- Values gradually converge toward true values

## On-Policy Control - SARSA (The Cautious Realist)

### **From State Values to Action Values**
**Problem:** Knowing $V(s)$ doesn't directly tell us what action to take

**Solution:** Learn $Q(s,a)$ values using TD approach

### **SARSA Algorithm**

**Name Origin:** **S**tate, **A**ction, **R**eward, **S**tate, **A**ction
- Complete experience tuple: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$

**SARSA Update Rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**TD Target:** $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$
- Uses actual next action $A_{t+1}$ that will be taken

### **Why SARSA is "On-Policy"**
**Key Characteristic:** Updates based on actions actually taken by current policy

**Implications:**
- If using $\varepsilon$-greedy policy, Q-values reflect exploration behavior
- Learns "realistic" values accounting for policy's exploration
- Conservative approach - factors in cost of exploration

**SARSA Backup Diagram:**
```
Q(S,A) ← R + γQ(S',A')
```
Simple linear chain from one state-action pair to next

### **The Cautious Philosophy**
SARSA answers: *"What is the value of my actions given that I will continue to explore?"*
- Realistic assessment including exploration costs
- Stable and safe learning
- May avoid optimal paths if exploration is risky

## Off-Policy Control - Q-Learning (The Bold Optimist)

### **The Key Innovation**
**Q-Learning separates:** 
- **Behavior policy:** How agent acts (e.g., $\varepsilon$-greedy)
- **Target policy:** Policy being learned (optimal greedy policy)

### **Q-Learning Update Rule**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**TD Target:** $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$

### **SARSA vs Q-Learning Target Comparison**

| Method | TD Target | Philosophy |
|--------|-----------|------------|
| **SARSA** | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | Use actual next action from policy |
| **Q-Learning** | $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ | Use best possible next action |

### **The Optimistic Philosophy**
Q-Learning answers: *"What is the value of the optimal policy, ignoring any exploration?"*
- Learns about perfect performance
- Bold approach - assumes optimal future behavior
- Can find truly optimal paths but may be unstable during learning

**Q-Learning Backup Diagram:**
```
Q(S,A) ← R + γ max_a Q(S',a)
```
Considers all possible next actions, takes maximum

## The Showdown - SARSA vs Q-Learning in Cliff Walking

### **Environment Setup**
```
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   | ← Top Row (Safe)
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   | G | ← Middle Row (Risky)
+---+---+---+---+---+---+---+---+---+---+---+---+
| S | C | C | C | C | C | C | C | C | C | C | C | ← Bottom Row (Cliff)
+---+---+---+---+---+---+---+---+---+---+---+---+
```

**Rules:**
- **Start:** S, **Goal:** G
- **Actions:** Up, Down, Left, Right
- **Rewards:** Each step = -1, Cliff = -100 (+ return to start)

**Two Possible Strategies:**
1. **Safe Path:** Top row → Goal (reward = -13)
2. **Optimal Path:** Middle row along cliff (reward = -11, but risky)

### **Results Comparison**

**Q-Learning Policy (Bold Optimist):**
```
+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+
| ↑ | → | → | → | → | → | → | → | → | → | → | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
| ↑ | C | C | C | C | C | C | C | C | C | C | C |
+---+---+---+---+---+---+---+---+---+---+---+---+
```
- **Strategy:** Risky path along cliff edge
- **Reasoning:** Learns optimal path ignoring exploration costs
- **Problem:** Performs poorly with exploration (frequent cliff falls)

**SARSA Policy (Cautious Realist):**
```
+---+---+---+---+---+---+---+---+---+---+---+---+
| → | → | → | → | → | → | → | → | → | → | → | ↓ |
+---+---+---+---+---+---+---+---+---+---+---+---+
| ↑ |   |   |   |   |   |   |   |   |   |   | G |
+---+---+---+---+---+---+---+---+---+---+---+---+
| ↑ | C | C | C | C | C | C | C | C | C | C | C |
+---+---+---+---+---+---+---+---+---+---+---+---+
```
- **Strategy:** Safe path through top row
- **Reasoning:** Learns cliff is too dangerous for exploring agent
- **Advantage:** Performs well in practice with exploration

### **Key Insights**

**When to Use Each:**

**Q-Learning (Off-Policy):**
- Want to learn optimal policy
- Can tolerate instability during learning
- Exploration costs are manageable
- Ultimate goal is optimal performance

**SARSA (On-Policy):**
- Want stable, safe learning
- Exploration is expensive/dangerous
- Need reliable performance during learning
- Operating in high-risk environments

## Summary: TD Learning Foundations

### **Core Contributions of TD Learning**

**Methodological Advances:**
1. **Step-by-step learning** - no waiting for episode completion
2. **Bootstrapping** - learn estimates from estimates
3. **Model-free** - no environment model required
4. **Online learning** - continuous improvement during interaction

**Key Algorithms:**

| Algorithm | Type | Update Rule | Best For |
|-----------|------|-------------|----------|
| **TD(0)** | Prediction | $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ | State value estimation |
| **SARSA** | On-Policy Control | $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$ | Safe, stable learning |
| **Q-Learning** | Off-Policy Control | $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)]$ | Optimal policy learning |

### **The Bias-Variance Trade-off**

| Method | Bias | Variance | Characteristics |
|--------|------|----------|----------------|
| **Monte Carlo** | Low | High | True returns, noisy signal |
| **TD(0)** | Higher | Lower | Estimated returns, stable signal |

**TD Learning Position:** Balances bias-variance trade-off through bootstrapping

### **Fundamental TD Concepts**

**TD Error:** $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
- **Interpretation:** Difference between prediction and reality
- **Role:** Learning signal that drives updates
- **Sign:** Positive = underestimated, Negative = overestimated

**Bootstrapping Benefits:**
- Faster learning than MC (learns from every step)
- More stable than pure model-free approaches
- Enables online learning in continuing tasks

## The Road Ahead - Future Directions

### **Current Limitations**

**1. One-Step Myopia**
- TD(0) only looks one step ahead
- **Question:** Can we look further without waiting for episode end?

**2. Tabular Representation**
- Current methods assume lookup tables for values
- **Problem:** Doesn't scale to large state spaces

### **Next Evolutionary Steps**

**1. n-Step Methods**
- Bridge between MC and TD learning
- Look n steps ahead: $R_{t+1} + \gamma R_{t+2} + ... + \gamma^n V(S_{t+n})$
- Tune bias-variance trade-off

**2. Planning and Learning Integration**
- Use experience to build internal models
- "Think" between moves using simulated experience
- Dyna-Q algorithm as example

**3. Function Approximation**
- Replace lookup tables with neural networks
- Enable learning in continuous and large state spaces
- Foundation for Deep Reinforcement Learning

### **The Scaling Challenge**
**Ultimate Goal:** Handle complex environments like:
- Chess/Go (discrete but huge state spaces)
- Robot control (continuous state spaces)
- Video games (high-dimensional observations)

**Solution Path:** Combine TD learning principles with powerful function approximators

## Key Formulas Reference

**TD(0) State Value Update:**
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**SARSA Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Q-Learning Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**TD Error:**
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

## Study Tips

1. **Understand bootstrapping concept** - learning estimates from estimates is key insight
2. **Practice TD error calculations** - core mechanism of all TD methods  
3. **Compare SARSA vs Q-Learning** - understand on-policy vs off-policy distinction
4. **Work through Cliff Walking** - concrete example of different philosophies
5. **See the progression** - MC → TD → Future methods build on these foundations
6. **Focus on the trade-offs** - bias vs variance, safety vs optimality, exploration vs exploitation