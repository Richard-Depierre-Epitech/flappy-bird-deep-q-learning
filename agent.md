# Deep Q-Learning Agent (`agent.py`)

This document provides a detailed explanation of `agent.py`, the main script for training and testing a Deep Q-Network (DQN) agent.

---

## 📜 Overview
The `agent.py` script implements a **reinforcement learning agent** that interacts with environments such as **Flappy Bird** and **CartPole**, using the **Deep Q-Learning (DQN) algorithm**. It supports:
- **Dueling DQN** for better action-value separation.
- **Double DQN** to reduce overestimation bias.
- **Experience Replay** for stable training.

The script can be run in **training mode** to learn an optimal policy or in **evaluation mode** to test a pre-trained model.

---

## 🚀 How to Run

### **1️⃣ Setup Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **2️⃣ Install Dependencies**
```bash
python3 -m pip install -r requirements.txt
```

### **3️⃣ Run the Agent**

#### **Training Mode**
To train the agent on a specific environment, use:
```bash
python3 agent.py cartpole --train
```
or
```bash
python3 agent.py flappybird --train
```
This will train a DQN agent in the specified environment and save the trained model.

#### **Evaluation Mode**
To test a pre-trained model, simply run:
```bash
python3 agent.py cartpole
```
or
```bash
python3 agent.py flappybird
```
Without the `--train` flag, the script will load the saved model and run it in evaluation mode.

#### **Exit Virtual Environment**
```bash
source deactivate
```

---

## 🔍 In-Depth Code Explanation

### **1️⃣ Class `Agent`**
The `Agent` class is responsible for initializing the environment, managing hyperparameters, setting up the DQN architecture, and handling the training process.

#### **Initialization (`__init__` Method)**
- Reads hyperparameters from `hyperparameters.yml`.
- Initializes the replay memory for experience storage.
- Creates both the **policy network** and the **target network**.
- Defines the optimizer and loss function for model training.
- Sets up paths for saving logs, model weights, and training graphs.

```python
with open('./hyperparameters.yml', 'r') as file:
    all_hyperparameters_sets = yaml.safe_load(file)
    hyperparameters = all_hyperparameters_sets[hyperparameter_set]
```
- Loads training settings from the YAML configuration file.

```python
self.policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
self.target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
```
- Initializes the **policy network** for decision-making and the **target network** for stable learning updates.

### **2️⃣ Training & Execution (`run` Method)**
This method controls both **training** and **evaluation** of the agent.

```python
def run(self, is_training, render=False):
```
- If `is_training=True`, the model is trained from scratch.
- If `render=True`, the environment displays the game visually.

#### **Environment Setup**
```python
env = gymnasium.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
```
- Loads either **Flappy Bird** or **CartPole** as the training environment.

#### **Training Loop**
```python
for episode in itertools.count():
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
```
- Runs episodes indefinitely until manually stopped.
- Converts the state observation into a tensor for model input.

#### **Action Selection**
```python
if (is_training and random.random() < epsilon):
    action = env.action_space.sample()
else:
    with torch.no_grad():
        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
```
- Uses an **ε-greedy policy**:
  - With probability **ε**, selects a **random action**.
  - Otherwise, selects the **best action** according to the Q-network.

#### **State Transition & Reward Processing**
```python
new_state, reward, terminated, _, info = env.step(action.item())
new_state = torch.tensor(new_state, dtype=torch.float, device=device)
reward = torch.tensor(reward, dtype=torch.float, device=device)
```
- Executes the action in the environment and retrieves the new state, reward, and termination signal.

#### **Experience Replay Storage**
```python
if (is_training):
    memory.append((state, action, new_state, reward, terminated))
```
- Stores the `(state, action, reward, next_state, done)` tuple into the replay memory for later training.

### **3️⃣ Model Optimization (`optimize` Method)**
```python
def optimize(self, mini_batch, policy_dqn, target_dqn):
```
- Samples a **mini-batch** of experiences from the replay memory.
- Computes **target Q-values** using either standard DQN or **Double DQN**.

```python
if self.enable_double_dqn:
    best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
    target_q = rewards + (1-terminations) * self.discount_factor_g * \
                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
else:
    target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
```
- If **Double DQN** is enabled, the **policy network** selects the best action, and the **target network** evaluates its value.
- Otherwise, the **maximum Q-value** from the target network is used.

```python
current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
loss = self.loss_fn(current_q, target_q)
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```
- **Computes loss** using Mean Squared Error (MSE).
- **Performs backpropagation** to update weights using **Adam optimizer**.

---

## 📊 Visual Explanations

The `images/` folder contains visual guides for understanding `agent.py`:
- **Double DQN Calculation (`double_DQN.png`)** → Explains how action selection is improved.
- **Policy Network (`policy_network_explanation.png`)** → Shows how states are processed into Q-values.
- **Optimization & Loss (`optimizer_explanation.png`)** → Illustrates how training is performed.

---

## 🔧 Future Improvements
- Implement **Prioritized Experience Replay**.
- Fine-tune hyperparameters for better convergence.
- Add more complex environments.

---

## 📝 License
This project is open-source and available for educational and research purposes.

🚀 **Happy Learning!** 🎮

