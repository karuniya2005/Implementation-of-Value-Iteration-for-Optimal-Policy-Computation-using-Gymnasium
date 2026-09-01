## Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

### Aim

To implement the Value Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, and to compute the optimal state-value function and optimal policy using the Bellman Optimality Equation.

### Problem Statement
Develop a Python program that applies the Value Iteration algorithm to the FrozenLake-v1 environment in Gymnasium. The program should iteratively update the value of each state until convergence and derive the optimal policy that maximizes the expected cumulative reward.

### Software Requirements
1. Python 3.10 or above
2. Gymnasium
3. NumPy
4. Matplotlib
5. Jupyter Notebook / Google Colab
   
### Environment Description
FrozenLake-v1 is a grid-world environment where an agent starts at the Start (S) state and attempts to reach the Goal (G) while avoiding Holes (H). The environment is stochastic when is_slippery=True, meaning the agent may not always move in the intended direction. Each state has four possible actions: Left, Down, Right, and Up.

### MDP Representation
An MDP is represented by the tuple:

(S, A, P, R, γ)

where:

S = Set of states
A = Set of actions
P = State transition probabilities
R = Reward function
γ = Discount factor

### Theory

Value Iteration is a Dynamic Programming algorithm used to compute the optimal value function of a Markov Decision Process. It repeatedly updates the value of each state using the Bellman Optimality Equation until the values converge.

Bellman Optimality Equation:
```
 V(s) = maxₐ ∑ₛ′ P(s′ | s, a) [ R(s, a, s′) + γV(s′) ]
```

After convergence, the optimal policy is obtained by selecting the action with the highest expected value for every state. This algorithm guarantees convergence to the optimal policy for finite MDPs.


### Algorithm
1. Initialize the value function of all states to zero.
2. Choose the discount factor γ and convergence threshold θ.
3. Compute the expected value of each action for every state.
4. Update the state value using the maximum action value.
5. Repeat the process until the maximum change in state values is less than θ.
6. Extract the optimal policy by selecting the action with the highest expected value for each state.
7. Display the optimal value function and policy.

### Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
env = env.unwrapped

gamma = 0.99
theta = 1e-8

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal
    value function, policy and number of iterations.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward +
                        gamma * V[next_state] * (not done)
                    )
            best_value = np.max(action_values)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
        iteration += 1
        if delta < theta:
            break
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward +
                    gamma * V[next_state] * (not done)
                )
        policy[s] = np.argmax(action_values)
    return V, policy, iteration

V, policy, iterations = value_iteration(
    env,
    gamma=gamma,
    theta=theta
)

print("Name: KARUNIYA M")
print("Register Number: 212223240068")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()    
```

### Output

#### Number of Iterations: 

<img width="332" height="95" alt="image" src="https://github.com/user-attachments/assets/428abccb-ba9b-456c-9ef3-e23e5928958f" />




#### Optimal State-Value Function:

<img width="396" height="132" alt="image" src="https://github.com/user-attachments/assets/e83803b5-8d23-4e34-9159-90dde8492233" />




#### Optimal Policy:

<img width="267" height="118" alt="image" src="https://github.com/user-attachments/assets/78a57831-8626-423a-a822-db4821f352d8" />



#### Ouput after changing the environment description:
```
for,
env_desc = [
    "SFFF",
    "FFHF",
    "FHFF",
    "HFFG"
]
]
```

<img width="373" height="373" alt="image" src="https://github.com/user-attachments/assets/fc8b4a83-2280-43c9-afc9-9510c6e2d40c" />


### Result
The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were obtained after convergence of the Bellman Optimality Equation.

### Inference
The Value Iteration algorithm successfully computed the optimal state-value function and policy for the given FrozenLake environment. By repeatedly applying the Bellman Optimality Equation until convergence, it identified the optimal action for each state based on the environment layout, maximizing the expected cumulative reward while accounting for state transitions and uncertainties.
