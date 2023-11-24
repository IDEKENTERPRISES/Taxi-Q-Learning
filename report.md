# CS4049 CA2
> By Hareeshan Elankeeran, Jules Desré-Crouch and Favour Jam


## Introduction

Reinforcement learning is the area of artificial intelligence that is focused on maximising a reward, through “mapping situations to actions”. The reward is maximised by balancing the exploration, or the information we can gather, and exploitation of an environment, or the information we know already. Reward maximisation to affect behaviour of an agent derived from experiments performed by both Ivan Pavlov and BF Skinner, with roots in psychology and neuroscience. The field was revitalised and integrated into computer science and artificial intelligence by Richard Sutton and Andrew Barto in 1979 and has had a steady influx of interest ever since. Reinforcement learning is centred around 3 core concepts, a policy $\pi$, which guides the agent on the behaviour it performs, a reward function $R\left(s\right)$, which allows for an agent to learn what a good and a bad decision are in the short term, and a value function $V\left(s\right)$, which is made up of the expected reward for the state.
A reinforcement agent can be modelled by a Markov decision process (MDP); a formalisation where the Markovian principle is followed when decisions are made. These decisions, or the linking of states via actions, is partly random, because of the environment, and partly made by the agent, due to exploration. The Markovian principle is the notion that for a given state, the value depends on the states before it. 
The Taxi environment, implemented using Gymnasium (formerly Open-AI Gym), was used due to the simplicity of both the environment implementation and the state-action table, which is made up of 500 observation states and 6 action states. The observation states are made up of each position on the $5\times5$ grid =\ 25 positions, each of the 5 locations the passenger can be at, which is either each of the 4 colours on the grid or in the taxi, and the destination being each of the 4 colours, $\therefore$ $25 \times 5 \times 4 = 500$. The 6 action states for the taxi agent are moving up, down, left, right, dropping off and picking up the passenger. 

## Epsilon-greedy

The epsilon-greedy method in the context of reinforcement learning, particularly in the Taxi environment from OpenAI's Gym, is a strategy used to balance exploration and exploitation. This method is crucial in learning optimal actions in an uncertain environment.

### Exploration-Exploitation Tradeoff:
- **Exploration** involves trying new actions to discover their effectiveness.
- **Exploitation** means using known actions that yield high rewards.

In the Taxi environment, where the goal is to navigate a taxi to pick up and drop off a passenger at specific locations:

- **Exploration** would be the taxi trying different paths, and pickup/dropoff actions to learn the best strategy.
- **Exploitation** would be the taxi using its learned knowledge to follow the most rewarding path and actions.

### Epsilon-Greedy Method:
1. **Parameter** $\epsilon$: Epsilon is a value between 0 and 1, determining the likelihood of exploration.
    - **High** $\epsilon$: More exploration, less exploitation.
    - **Low** $\epsilon$: More exploitation, less exploration.

1. **Action Selection**:
- With probability $\epsilon$, the agent (taxi) chooses an action at random (exploration).
- With probability 1−$\epsilon$, the agent chooses the best-known action (exploitation).

### Application in Taxi Environment:
- At the start, the taxi might randomly explore different actions (pickups, drop-offs, movements) to understand the environment.
- As it learns, it starts exploiting known paths and actions that lead to successful passenger delivery.
- Over time, the value of $\epsilon$ can be decreased, gradually shifting from exploration to exploitation.
### Balancing with Epsilon Decay:
- Often, $\epsilon$ is decayed over time to slowly shift the focus from exploration to exploitation as the taxi gathers more experience.
### Importance in Taxi Environment:
- Essential for learning the best routes and actions in the grid.
- Prevents the agent from getting stuck in suboptimal strategies.

In summary, the epsilon-greedy method in the Taxi environment helps the taxi learn from both new experiences (exploration) and past successes (exploitation), balancing the need to discover new strategies and the need to follow known, rewarding paths.


## Implementation of Epsilon-greedy

To design and train a neural network agent using the Epsilon-greedy method for the 'Taxi-v3' environment from Gymnasium, the following approach was taken:

### Agent Design

1. **Environment Setup**: 
   - The 'Taxi-v3' environment is a discrete action space game where the taxi must pick up and drop off passengers at different locations.
   - The environment was loaded into TensorFlow using `suite_gym.load`.

2. **Neural Network Architecture**:
   - A Q-network (`q_network.QNetwork`) was used, appropriate for value-based methods in reinforcement learning.
   - The network was designed with fully connected layers (60 and 10 neurons respectively), specified in `fc_layer_params`.

3. **Agent Configuration**:
   - A DQN (Deep Q-Network) agent (`dqn_agent.DqnAgent`) was implemented.
   - The optimizer used was Adam with a learning rate (ALPHA) of 0.001.
   - The training step counter was initialized to track the number of training steps.

4. **Policies**:
   - Two main policies were created: `eval_policy` for evaluation and `collect_policy` for data collection.
   - An Epsilon-greedy policy (`py_epsilon_greedy_policy.EpsilonGreedyPolicy`) was used for exploration, with EPSILON set to 0.6.

5. **Replay Buffer**:
   - A replay buffer (`tf_uniform_replay_buffer.TFUniformReplayBuffer`) was used to store experiences.

6. **Training Setup**:
   - A dynamic step driver (`dynamic_step_driver.DynamicStepDriver`) was used to collect data.
   - A dataset was created from the replay buffer for training the agent.
   - Training involved updating the agent's network parameters based on sampled experiences.

### Motivation Behind Design Choices

- **Q-Network**: Chosen for its effectiveness in value-based reinforcement learning, particularly in discrete action spaces.
- **Adam Optimizer**: Known for its efficiency and adaptive learning rate properties.
- **Epsilon-Greedy Policy**: Strikes a balance between exploration (trying new actions) and exploitation (using known information). The choice of EPSILON value is crucial for this trade-off.
- **Fully Connected Layers**: Suitable for the relatively simple input space of the 'Taxi-v3' environment.
- **Replay Buffer**: Enables the learning from past experiences, making the training process more stable and efficient.

### Parameters and Adjustments

- `ALPHA (Learning Rate)`: Set to 0.001, balancing the speed of convergence and the stability of the learning process.
- `EPSILON (Exploration Rate)`: Initially set to 0.6, indicating a higher preference for exploration at the beginning.

### Training Details

- The agent was trained over 10,000 iterations.
- The average return for each policy was computed to evaluate performance.
- Losses were logged every 200 steps to monitor the training progress.

### Observations from Training

- The loss decreased steadily initially, indicating learning and improvement in policy.
- However, after 8400 steps, there was a dramatic increase in loss, suggesting a possible issue in the training process, such as catastrophic forgetting or instability in the learning algorithm.
- The final computed average return was -200, indicating poor performance of the agent.

### Possible Issues and Recommendations

- **Exploding Loss**: The drastic increase in loss could be due to several factors, including parameter settings or instability in the Q-learning updates. Investigating learning rate, replay buffer size, and network architecture could be beneficial.
- **Negative Reward**: The -200 average return suggests the agent's policy did not learn effectively. Tuning the balance between exploration and exploitation, and considering other factors like different network architectures, might help.

### Figures and Visualization

To further elucidate the training process and the performance of the agent, visualizations such as plots of loss over time and average returns at different training stages would be beneficial. However, the current setup does not include code for generating these plots. 

### Acknowledgements

The code utilizes TensorFlow Agents (TF-Agents), a library for reinforcement learning in TensorFlow, and Gymnasium for the environment setup. The use of these libraries greatly simplifies the implementation of complex reinforcement learning algorithms and environments.
