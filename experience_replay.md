# Experience Replay (`experience_replay.py`)

This document provides an in-depth explanation of `experience_replay.py`, which implements the **Experience Replay Memory** used in reinforcement learning.

---

## 📜 Overview
The `experience_replay.py` file defines a **Replay Memory buffer** that stores past experiences (state, action, reward, next state, done) and allows the agent to sample from them during training.

**Why is Experience Replay important?**
- **Breaks correlation** between consecutive experiences, making training more stable.
- **Improves sample efficiency** by reusing past experiences.
- **Allows mini-batch updates**, reducing variance in training.

This mechanism is widely used in **Deep Q-Networks (DQN)** to stabilize learning.

---

## 🚀 How It Works

### **1️⃣ Importing Dependencies**
```python
from collections import deque
import random
```
- Uses `deque` (a double-ended queue) for efficient memory management.
- Uses `random` for **sampling batches** of experiences.

### **2️⃣ Class `ReplayMemory`**
```python
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque(maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
```
#### **Initialization (`__init__` method)**
- `maxlen`: Sets the maximum buffer size (old experiences are removed when full).
- `seed`: (Optional) Ensures reproducibility when sampling experiences.
- Uses a **deque** (FIFO queue) for efficient removal of old experiences.

### **3️⃣ Storing Experiences**
```python
def append(self, transition):
    self.memory.append(transition)
```
- Adds a new experience (`transition`) to the memory.
- A **transition** is a tuple `(state, action, reward, next_state, done)`.

### **4️⃣ Sampling a Batch**
```python
def sample(self, sample_size):
    return random.sample(self.memory, sample_size)
```
- Returns a **random mini-batch** of size `sample_size`.
- Sampling randomly **breaks correlation** between consecutive experiences, making training more effective.

### **5️⃣ Checking Memory Size**
```python
def __len__(self):
    return len(self.memory)
```
- Returns the current number of experiences stored in memory.
- Useful for ensuring the buffer has enough samples before training starts.

---

## 📊 Key Features
1. **Fixed-size memory buffer** (removes old experiences when full).
2. **Efficient storage using `deque`** (constant-time appends/removals).
3. **Random sampling** to improve learning stability.
4. **Reproducibility** with optional seeding.

---

## 📷 Visual Explanation

The `images/` folder contains diagrams illustrating Experience Replay in action:
- **Replay Memory Mechanism** → Shows how experiences are stored and sampled.

---

## 🔧 Future Improvements
- Implement **Prioritized Experience Replay** to sample important experiences more frequently.
- Optimize sampling efficiency for large datasets.

---

## 📝 License
This project is open-source and available for educational and research purposes.

🚀 **Happy Learning!** 🎮