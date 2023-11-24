# CS4049 CA2
> By Hareeshan Elankeeran, Jules Desré-Crouch and Favour Jam


## Introduction

Reinforcement learning is the area of artificial intelligence that is focused on maximising a reward, through “mapping situations to actions”. The reward is maximised by balancing the exploration, or the information we can gather, and exploitation of an environment, or the information we know already. Reward maximisation to affect behaviour of an agent derived from experiments performed by both Ivan Pavlov and BF Skinner, with roots in psychology and neuroscience. The field was revitalised and integrated into computer science and artificial intelligence by Richard Sutton and Andrew Barto in 1979 and has had a steady influx of interest ever since. Reinforcement learning is centred around 3 core concepts, a policy $\pi$, which guides the agent on the behaviour it performs, a reward function $R(s)$, which allows for an agent to learn what a good and a bad decision are in the short term, and a value function $V(s)$, which is made up of the expected reward for the state.
A reinforcement agent can be modelled by a Markov decision process (MDP); a formalisation where the Markovian principle is followed when decisions are made. These decisions, or the linking of states via actions, is partly random, because of the environment, and partly made by the agent, due to exploration. The Markovian principle is the notion that for a given state, the value depends on the states before it. 
The Taxi environment, implemented using Gymnasium (formerly Open-AI Gym), was used due to the simplicity of both the environment implementation and the state-action table, which is made up of 500 observation states and 6 action states. The observation states are made up of each position on the $5\times5$ grid = $25$ positions, each of the 5 locations the passenger can be at, which is either each of the 4 colours on the grid or in the taxi, and the destination being each of the 4 colours, $\therefore$ $25 \times 5 \times 4 = 500$. The 6 action states for the taxi agent are moving up, down, left, right, dropping off and picking up the passenger. 

The epsilon-greedy method is a strategy used to balance exploration and exploitation. This method is crucial in learning optimal actions in an uncertain environment.

**Exploration** involves trying new actions to discover their effectiveness.
**Exploitation** means using known actions that yield high rewards.

In the Taxi environment, where the goal is to navigate a taxi to pick up and drop off a passenger at specific locations, exploration would be the taxi trying different paths, and pickup/dropoff actions to learn the best strategy, whilst exploitation would be the taxi using its learned knowledge to follow the most rewarding path and actions. The reward factors are $-1$ for each step, $+20$ for delivering the passenger, and $-10$ for executing pickup + drop off actions. 
At the start, the taxi may randomly explore different actions (pickups, drop-offs, movements) to understand the environment. As it learns, it starts exploiting known paths and actions that lead to successful passenger delivery. Often, $\epsilon$ is generally decayed over time to slowly shift the focus from exploration to exploitation as the taxi gathers more experience. By both exploring and exploiting the environment, the agent learns the best routes and actions in the grid, whilst also preventing the agent from getting stuck in a suboptimal strategy.


## Implementation of Epsilon-greedy

A deep neural network is a form of artificial neural network which contains an increased amount of weight nodes, inside hidden layers. These hidden layers are in between the input and output layers. By combining this approach with a Q learning algorithm, deep Q learning can be achieved, which allows for a wider application for many other domains. [^1] To design and train a neural network agent using the Epsilon-greedy method for the 'Taxi-v3' environment from Gymnasium, the following approach was taken:

### Agent Design

1. **Environment Setup**: 
   - We loaded the environment into TensorFlow using  using `suite_gym.load`, a TensorFlow wrapper for the OpenAI gym models.
   - We defined hyperparameters like the learning rate $\alpha$, the exploration-exploitation threshold $\epsilon$ and the number of iterations. We also defined the layout of the neural network, in a 6 -> 500 -> 6 formation. This was because the number of possible actions was 6, and the number of observable in the environment was 500.

2. **Neural Network Architecture**:
   - A Q-network (`q_network.QNetwork`) was used. This combines the Q based learning approach with the use of deep neural networks. This is commonly used for value-based methods in reinforcement learning.[^2]. 

3. **Agent Configuration**:
   - We implemented a DQN (Deep Q-Network) agent (`dqn_agent.DqnAgent`), using the TF-agents package, and added the Adam optimizer, with a learning rate ($\alpha$) of 0.001.
   - The training step counter was initialized to track the number of training steps. This could allow us to plot our losses. 

4. **Policies**:
   - The agent was made up of two policies, the evaluation policy, and the collect policy. The evaluation policy was the main policy of the agent, whilst the collect policy, was used to collect any data that was gathered by the agent.

5. **Replay Buffer**:
   - A replay buffer (`tf_uniform_replay_buffer.TFUniformReplayBuffer`) was used to store experiences.

6. **Training Setup**:
   - A dynamic step driver (`dynamic_step_driver.DynamicStepDriver`) was used to collect data.
   - A dataset was created from the replay buffer for training the agent.
   - Training involved updating the agent's network parameters based on the sampled experiences. The agent would look at the batch size as a Markovian state. 
   - The agent was trained over 10,000 iterations. The average return for each policy was computed to evaluate performance. Losses were logged every 200 steps to monitor the training progress.

### Conclusion:

- The loss decreased steadily initially, indicating learning and improvement in policy.
- However, after 8400 steps, there was a dramatic increase in loss, suggesting a possible issue in the training process, such as instability in the learning algorithm, or a problem with the training. This also could have been because of the hyperparameters given at the start, as well as the epsilon not decaying and forcing the agent towards exploitation of the environment. The final computed average return was -200. 

### Figures and Visualization



### Acknowledgements

The code utilizes TensorFlow Agents (TF-Agents), a library for reinforcement learning in TensorFlow, and Gymnasium for the environment setup. The use of these libraries greatly simplifies the implementation of complex reinforcement learning algorithms and environments.

### References

[^1]: Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature 2015 518:7540, 518(7540), 529–533. https://doi.org/10.1038/nature14236

[^2]: B. Jang, M. Kim, G. Harerimana and J. W. Kim, "Q-Learning Algorithms: A Comprehensive Classification and Applications," in IEEE Access, vol. 7, pp. 133653-133667, 2019, doi: 10.1109/ACCESS.2019.2941229.



