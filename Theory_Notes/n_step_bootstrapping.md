# Reinforcement Learning: n-Step Bootstrapping
## Complete Study Notes

## The Spectrum of Learning - Beyond Two Extremes

### **The Current Landscape**

We now have two powerful but contrasting model-free learning approaches:

| **Monte Carlo Learning** | **One-Step TD Learning** |
|---------------------------|--------------------------|
| **Philosophy:** Super-patient learner | **Philosophy:** Myopic, impatient learner |
| **Target:** $G_t$ (true return) | **Target:** $R_{t+1} + \gamma V(S_{t+1})$ |
| **Bias:** Low (unbiased) | **Bias:** Higher (biased by estimates) |
| **Variance:** High (noisy signal) | **Variance:** Low (stable updates) |
| **Learning Speed:** Slow (end of episode) | **Learning Speed:** Fast (every step) |

### **The Central Question**
> **Must we choose between these two extremes?** Can we find middle ground between waiting for the absolute end and looking only one step ahead?

### **The Weather Forecaster Analogy**

**Monte Carlo Forecaster:** 
- Predicts full week on Monday
- Waits until Sunday to use complete real data
- **Problem:** Freak Saturday heatwave unfairly penalizes Monday forecast

**1-Step TD Forecaster:**
- Updates Monday prediction using only Tuesday's real weather + new forecast
- **Problem:** Relies heavily on another forecast (still just a guess)

**n-Step Forecaster:**
- Waits until Thursday morning
- Uses 3 days of real weather + updated forecast for remaining days  
- **Advantage:** More real information than 1-step, doesn't wait for full episode

This intuition leads us to **n-step bootstrapping** - a unified framework that bridges MC and TD methods.

## The n-Step Return - Mixing Reality with Expectation

### **Formalizing the Spectrum**

**The n-Step Target Return:**
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

**Compact Form:**
$$G_{t:t+n} = \sum_{k=1}^{n} \gamma^{k-1} R_{t+k} + \gamma^n V(S_{t+n})$$

### **Special Cases**

| n Value | Method | Target Formula | Characteristics |
|---------|--------|----------------|-----------------|
| **n = 1** | 1-Step TD | $R_{t+1} + \gamma V(S_{t+1})$ | Pure bootstrapping |
| **n = 2** | 2-Step TD | $R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$ | Mixed approach |
| **n → ∞** | Monte Carlo | $\sum_{k=1}^{T-t} \gamma^{k-1} R_{t+k}$ | Pure sampling |

### **Backup Diagrams Spectrum**

**1-Step TD:**
```
S_t → S_{t+1} (bootstrapped)
```

**2-Step TD:**  
```
S_t → S_{t+1} → S_{t+2} (bootstrapped)
```

**n-Step TD:**
```
S_t → S_{t+1} → ... → S_{t+n} (bootstrapped)
```

**Monte Carlo:**
```
S_t → S_{t+1} → ... → Terminal
```

### **n-Step TD(0) Update Rule**
$$V(S_t) \leftarrow V(S_t) + \alpha [G_{t:t+n} - V(S_t)]$$

## Worked Example: Random Walk Efficiency

### **Environment Setup**
```
[Term 0] ←→ A ←→ B ←→ C ←→ D ←→ E ←→ [Term +1]
```

**Configuration:**
- **Rewards:** Right terminal = +1, all others = 0
- **Discount:** $\gamma = 1$, **Learning rate:** $\alpha = 0.1$
- **Initial values:** $V(s) = 0.5$ for all non-terminal states
- **True values:** $v(A) = 1/6, v(B) = 2/6, v(C) = 3/6, v(D) = 4/6, v(E) = 5/6$

### **Episode Trajectory Analysis**
**Successful path:** $C \rightarrow D \rightarrow E \rightarrow \text{Term}(+1)$

**Update Target Calculations for $V(C)$:**

**1-Step Return (n=1):**
$$G_{t:t+1} = R_{t+1} + \gamma V(S_{t+1}) = 0 + 1 \times V(D) = 0.5$$
$$V(C) \leftarrow 0.5 + 0.1 \times (0.5 - 0.5) = 0.5$$
**Result:** No learning (neighbor still uninformative)

**2-Step Return (n=2):**
$$G_{t:t+2} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) = 0 + 1 \times 0 + 1^2 \times V(E) = 0.5$$
$$V(C) \leftarrow 0.5 + 0.1 \times (0.5 - 0.5) = 0.5$$
**Result:** Still no learning

**3-Step Return (n=3):**
$$G_{t:t+3} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 V(\text{Term})$$
$$= 0 + 1 \times 0 + 1^2 \times 1 + 1^3 \times 0 = 1$$
$$V(C) \leftarrow 0.5 + 0.1 \times (1 - 0.5) = 0.55$$
**Result:** Successful learning! Sees real reward signal

### **Key Insight: Rapid Credit Propagation**
- **1-step & 2-step:** Cannot see past uninformative initial estimates
- **3-step:** Long enough to capture real reward information
- **Advantage:** Information propagates much faster than gradual 1-step "trickling"

## n-Step Control - SARSA with Longer Memory

### **Extending to Action-Values**

**n-Step SARSA Target:**
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})$$

**Update Rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_{t:t+n} - Q(S_t, A_t)]$$

### **On-Policy Nature Preserved**
- **Key characteristic:** Uses actual action $A_{t+n}$ chosen by policy
- **Implication:** Still learns realistic values accounting for exploration
- **Benefit:** Less myopic than 1-step SARSA while maintaining stability

### **Rapid Credit Assignment Example**

**10-Step Successful Path:**
$$S_0 \xrightarrow{A_0} S_1 \xrightarrow{A_1} S_2 \rightarrow ... \rightarrow S_9 \xrightarrow{A_9} \text{Goal}$$

**1-Step SARSA:**
- Only updates $Q(S_9, A_9)$ immediately
- Information trickles back one step per episode
- Requires many episodes for full credit assignment

**10-Step SARSA:**
- Updates $Q(S_0, A_0)$ using full 10-step return
- **Result:** Entire successful path gets credit simultaneously
- **Metaphor:** "Lightning illuminates entire path" vs "slow trickle"

## The Off-Policy Conundrum

### **The Challenge**
**Problem:** Agent follows behavior policy $b$ but wants to learn about target policy $\pi$

**Simple extension doesn't work:**
- Take $n$ steps with behavior policy $b$  
- Use max operation only at the end
- **Issue:** Ignores that middle $n-1$ actions were also off-policy

### **Two Solution Approaches**

## Approach 1: Importance Sampling

### **The Correction Factor Method**

**Core Idea:** Weight the n-step experience by how likely it would be under target policy

**Importance Sampling Ratio:**
$$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

**Corrected Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \rho_{t:t+n-1} [G_{t:t+n} - Q(S_t, A_t)]$$

### **How It Works**
- **$\pi(A_k|S_k)$:** Probability target policy would take action $A_k$
- **$b(A_k|S_k)$:** Probability behavior policy actually took action $A_k$
- **If any $\pi(A_k|S_k) = 0$:** Entire ratio $\rho = 0$ (experience discarded)
- **If any $b(A_k|S_k)$ is small:** Ratio explodes → high variance

### **Advantages and Disadvantages**

| Advantages | Disadvantages |
|------------|---------------|
| Theoretically correct | Extremely high variance |
| Unbiased estimates | Unstable learning |
| Conceptually straightforward | Can slow learning significantly |

## Approach 2: Tree-Backup Algorithm

### **The Elegant Alternative**

**Core Idea:** Avoid importance sampling by mixing samples with expectations at every step

**Key Innovation:** Instead of correcting a sampled path, construct a mixed target that naturally handles off-policy learning

### **Tree-Backup Target Construction**

**At each step $k$ from $t$ to $t+n-1$:**
1. **Follow sampled action:** $A_k$ (the "spine" of the tree)
2. **Add expected values:** For all other actions $a \neq A_k$, weighted by $\pi(a|S_k)$

**Mathematical Formulation:**
$$G_{t:t+n}^{\text{TB}} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) + \gamma \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n}^{\text{TB}}$$

### **Backup Diagram Visualization**
```
Tree structure showing:
- Solid spine: Actually sampled actions
- Dashed branches: Expected values of unsampled actions  
- Each state branches to all possible actions
- Only one action per state is actually sampled
- Others contribute via expectation weighted by π(a|s)
```

### **Tree-Backup Advantages**

| Property | Benefit |
|----------|---------|
| **No importance sampling** | Low variance, stable updates |
| **Full off-policy capability** | Learns optimal policy while exploring |
| **Natural action weighting** | Smoothly handles stochastic policies |
| **Computational efficiency** | No explosive correction factors |

## Algorithm Comparison Summary

### **Complete Method Spectrum**

| Method | Policy Type | n-Steps | Variance | Stability | Use Case |
|--------|-------------|---------|----------|-----------|----------|
| **n-Step SARSA** | On-Policy | ✓ | Medium | High | Safe, stable learning |
| **n-Step + Importance Sampling** | Off-Policy | ✓ | Very High | Low | Theoretical interest |
| **Tree-Backup** | Off-Policy | ✓ | Medium | High | Practical off-policy learning |

### **Bias-Variance Trade-off Control**

**The n-Parameter as a Dial:**
- **Small n:** Lower variance, higher bias (more bootstrapping)
- **Large n:** Higher variance, lower bias (more real samples)  
- **Optimal n:** Problem-dependent, often somewhere in middle

## The Scaling Challenge Ahead

### **Current Limitation: The Lookup Table**

**Fundamental Assumption:** All methods so far assume tabular representation
- Store $V(s)$ or $Q(s,a)$ for every state/action pair
- **Works for:** Small problems (grid worlds, tic-tac-toe)

**Real-World Problem Sizes:**
- **Chess:** ~$10^{47}$ states
- **Go:** ~$10^{170}$ states  
- **Robotics:** Infinite continuous states
- **Vision-based control:** Astronomical pixel combinations

### **The Generalization Imperative**

**Critical Question:** *How can an agent make good decisions in states it has never seen?*

**Answer:** Replace lookup tables with **function approximators**
- Learn parameters $\mathbf{w}$ of function $\hat{v}(s, \mathbf{w})$ or $\hat{q}(s,a, \mathbf{w})$
- **Options:** Linear functions, neural networks, etc.
- **Goal:** Generalize from limited experience to unseen states

### **The Next Great Leap**
**From:** Tabular methods solving toy problems  
**To:** Approximate methods tackling real-world complexity

This transition represents the bridge from academic exercises to practical AI systems that can handle the immense complexity of real environments.

## Key Formulas Reference

**n-Step Return:**
$$G_{t:t+n} = \sum_{k=1}^{n} \gamma^{k-1} R_{t+k} + \gamma^n V(S_{t+n})$$

**n-Step TD Update:**
$$V(S_t) \leftarrow V(S_t) + \alpha [G_{t:t+n} - V(S_t)]$$

**n-Step SARSA Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_{t:t+n} - Q(S_t, A_t)]$$

**Importance Sampling Ratio:**
$$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

**Off-Policy n-Step Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \rho_{t:t+n-1} [G_{t:t+n} - Q(S_t, A_t)]$$

## Study Tips

1. **Understand the spectrum concept** - MC and TD are endpoints, not separate methods
2. **Practice n-step calculations** - work through different values of n on simple examples
3. **Visualize backup diagrams** - see how information flows differently with various n
4. **Grasp the credit assignment advantage** - why longer lookahead accelerates learning
5. **Compare off-policy solutions** - understand trade-offs between importance sampling and tree-backup
6. **Recognize the scaling limitation** - tabular methods hit fundamental barriers
7. **See the progression** - n-step methods bridge to function approximation era
