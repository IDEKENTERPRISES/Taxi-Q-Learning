# CS4049 CA2
> By Hareeshan Elankeeran, Jules Desré-Crouch and Favour Jam

## Epsilon-greedy

The epsilon-greedy method in the context of reinforcement learning, particularly in the Taxi environment from OpenAI's Gym, is a strategy used to balance exploration and exploitation. This method is crucial in learning optimal actions in an uncertain environment.

### Exploration-Exploitation Tradeoff:
- **Exploration** involves trying new actions to discover their effectiveness.
- **Exploitation** means using known actions that yield high rewards.

In the Taxi environment, where the goal is to navigate a taxi to pick up and drop off a passenger at specific locations:

- **Exploration** would be the taxi trying different paths, and pickup/dropoff actions to learn the best strategy.
- **Exploitation** would be the taxi using its learned knowledge to follow the most rewarding path and actions.

### Epsilon-Greedy Method:
1. **Parameter** ϵ: Epsilon is a value between 0 and 1, determining the likelihood of exploration.
    - **High** ϵ: More exploration, less exploitation.
    - **Low** ϵ: More exploitation, less exploration.

1. **Action Selection**:
- With probability ϵ, the agent (taxi) chooses an action at random (exploration).
- With probability 1−ϵ, the agent chooses the best-known action (exploitation).

### Application in Taxi Environment:
- At the start, the taxi might randomly explore different actions (pickups, drop-offs, movements) to understand the environment.
- As it learns, it starts exploiting known paths and actions that lead to successful passenger delivery.
- Over time, the value of ϵ can be decreased, gradually shifting from exploration to exploitation.
### Balancing with Epsilon Decay:
- Often, ϵ is decayed over time to slowly shift the focus from exploration to exploitation as the taxi gathers more experience.
### Importance in Taxi Environment:
- Essential for learning the best routes and actions in the grid.
- Prevents the agent from getting stuck in suboptimal strategies.

In summary, the epsilon-greedy method in the Taxi environment helps the taxi learn from both new experiences (exploration) and past successes (exploitation), balancing the need to discover new strategies and the need to follow known, rewarding paths.