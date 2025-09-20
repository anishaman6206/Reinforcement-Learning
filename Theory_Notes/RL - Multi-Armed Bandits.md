# Reinforcement Learning: Multi-Armed Bandits
## Complete Study Notes

---

## 🎯 **Part 1: The Big Idea - What Makes RL Different?**

### **Supervised Learning vs Reinforcement Learning**

| **Supervised Learning** | **Reinforcement Learning** |
|-------------------------|----------------------------|
| **Feedback Type:** Instructive | **Feedback Type:** Evaluative |
| "This is the right answer (5+5=10)" | "That action scored 7/10" |
| Teacher shows correct solution | Critic gives performance score |
| **No exploration needed** | **Must actively search for good actions** |

### **Key Insight**
> **RL learns from evaluation, not instruction.** The agent must discover the best actions through trial and error.

---

## 🎰 **The k-Armed Bandit Problem**

### **The Casino Analogy**
- **Scenario:** Walk into casino with k slot machines (arms)
- **Resources:** Limited tokens (e.g., 1000)
- **Hidden Information:** Each machine has different payout probability
- **Goal:** Maximize total winnings

### **Mathematical Formulation**

**Key Terms:**
- **Action (a):** Choice of which lever to pull
- **A_t:** Action chosen at time step t
- **R_t:** Reward received at time step t
- **q*(a):** True mean reward for action a

**True Value Formula:**
```
q*(a) = E[R_t | A_t = a]
```
*"The expected average reward for taking action a"*

**The Challenge:** You don't know q*(a) values - must discover them!

---

## ⚖️ **The Core Conflict: Exploration vs Exploitation**

### **The Dilemma**

**🎯 Exploitation**
- Stick with currently best-known machine
- Maximize immediate reward
- Safe but potentially suboptimal

**🔍 Exploration**
- Try different/unknown machines
- Gather information for future decisions
- Risk immediate reward for potential long-term gain

### **The Trade-off**
> **Cannot explore and exploit simultaneously** - Every token spent exploring gives up guaranteed reward from current best option.

---

## 🚫 **Part 2: The Greedy Strategy (And Why It Fails)**

### **How Greedy Works**

**Phase 1: Estimate**
```
Q_t(a) = Sum of rewards for action a / Number of times action a was taken
```

**Phase 2: Exploit**
```
A_t = argmax_a Q_t(a)
```
*"Always pick the action with highest estimated value"*

### **Fatal Flaw: The Greedy Trap**

**Example Scenario:**
- 3 machines with true values: q*(1)=1, q*(2)=5, q*(3)=10
- Agent tries each once: R₁=1, R₂=7 (lucky!), R₃=4 (unlucky!)
- Estimates: Q(1)=1, Q(2)=7, Q(3)=4
- **Result:** Agent forever chooses Machine 2, never discovers Machine 3 is best!

**Problem:** Gets stuck on first "good enough" option, never looks back.

---

## ✨ **Part 3: ε-Greedy Method - The Smart Solution**

### **The Big Idea**
> **"Be greedy most of the time, but occasionally do something random"**

### **The Algorithm**
At each time step:
1. Generate random number between 0 and 1
2. **If number > ε:** Exploit (choose best known action)
3. **If number ≤ ε:** Explore (choose random action)

**Typical values:** ε = 0.1 (10% exploration, 90% exploitation)

### **Why It Works**
- **Prevents getting stuck:** Regular random exploration
- **Converges to optimal:** Law of Large Numbers ensures Q_t(a) → q*(a)
- **Balances trade-off:** Mostly exploits, but keeps exploring

### **Incremental Update Formula**
When choosing action A and receiving reward R:

```
N(A) = N(A) + 1                    // Increment count
Q(A) = Q(A) + (1/N(A)) × [R - Q(A)]  // Update estimate
```

**Key Insight:** Step size 1/N(A) decreases as we sample more → estimates become more stable

---

## 📊 **Part 4: Step-by-Step Example**

### **Setup**
- **Game:** 3-armed bandit
- **True values:** q*(1)=2, q*(2)=6, q*(3)=4
- **Strategy:** ε-Greedy with ε=0.1
- **Initial state:** Q=(0,0,0), N=(0,0,0)

### **Game Progression**

| Step | Decision Logic | Action | Reward | Updated Brain |
|------|---------------|--------|---------|---------------|
| 1 | All Qs=0, pick randomly | Arm 1 | R=1 | Q=(1,0,0) |
| 2 | Exploit: max is Arm 1 | Arm 1 | R=3 | Q=(2,0,0) |
| 3 | **Explore:** Random pick | Arm 3 | R=5 | Q=(2,0,5) |
| 4 | Exploit: max is Arm 3 | Arm 3 | R=3 | Q=(2,0,4) |
| 5 | Exploit: max is Arm 3 | Arm 3 | R=6 | Q=(2,0,4.67) |
| 6 | **Explore:** Random pick | Arm 2 | R=8 | Q=(2,8,4.67) |
| 7 | Exploit: max is Arm 2 | Arm 2 | R=5 | Q=(2,6.5,4.67) |

**Final estimates converging toward truth:** Q=(2, 6.5, 4.67) ≈ q*=(2, 6, 4)

---

## 🧠 **Part 5: Advanced Exploration Strategies**

### **1. Optimistic Initial Values**

**Concept:** Start with unrealistically high Q values

**How it works:**
- Initialize: Q₁(a) = 20 for all actions (if max reward is 10)
- After first try: Real reward (e.g., 5) makes Q value drop
- Untried arms still look "amazing" → natural exploration

**Benefit:** Tricks purely greedy agent into exploring everything

### **2. Upper-Confidence-Bound (UCB)**

**Formula:**
```
A_t = argmax_a [Q_t(a) + c × √(ln(t)/N_t(a))]
```

**Components:**
- **Q_t(a):** Exploitation term (current value estimate)
- **c × √(ln(t)/N_t(a)):** Exploration bonus (uncertainty term)
  - **t:** Total number of pulls
  - **N_t(a):** Times action a was tried
  - **c:** Exploration parameter

**Smart Exploration Logic:**
- High Q_t(a) + Low N_t(a) = High UCB score
- Prioritizes actions that are both promising AND uncertain

---

## 🌟 **Part 6: Why This Matters - Real-World Applications**

### **Direct Applications**

**A/B Testing**
- **Actions:** Different website designs
- **Rewards:** Click-through rates
- **Goal:** Find best design while maximizing engagement

**Clinical Trials**
- **Actions:** Different treatments
- **Rewards:** Patient outcomes
- **Goal:** Balance giving current best treatment vs. testing new ones

### **Bridge to Full Reinforcement Learning**

**Current Problem:** Non-associative (best action always same)

**Next Level:** Associative/Contextual
- Best action depends on situation/state
- Self-driving car: action depends on traffic conditions
- Game AI: move depends on board state

**Methods Transfer:** ε-greedy, UCB used as components in complex RL systems

---

## 📚 **Key Takeaways & Summary**

### **Core Challenge**
> **Exploration-Exploitation Dilemma:** The fundamental trade-off in all reinforcement learning

### **Solution Strategies**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Greedy** | Always pick best known | Simple, fast convergence to local optimum | Gets stuck, misses global optimum |
| **ε-Greedy** | Random exploration with probability ε | Simple, guarantees convergence | Exploration is random/unfocused |
| **Optimistic Initial** | Start with high value estimates | Encourages early exploration | Only helps at beginning |
| **UCB** | Explore uncertain promising actions | Smart, directed exploration | More complex computation |

### **Universal Principle**
**Success in RL requires balancing:**
- **Information gathering** (exploration)
- **Reward maximization** (exploitation)

*This fundamental insight applies from simple bandits to complex AI systems navigating real-world environments.*

---

## 💡 **Study Tips**

1. **Remember the casino analogy** - makes abstract concepts concrete
2. **Focus on the trade-off** - every RL algorithm addresses exploration vs exploitation
3. **Understand why greedy fails** - shows necessity of exploration
4. **Practice the math** - incremental updates are used everywhere in RL
5. **See the bigger picture** - bandits are building blocks for full RL systems