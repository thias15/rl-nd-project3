import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 8e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_STEP = 1         # Step size for updating the networks
UPDATE_ITER = 1         # Number of updates per step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using: %s' %(device))

torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# non-determinism is unavoidable in some functions that use atomicAdd for example
# such as torch.nn.functional.embedding_bag() and torch.bincount()
        
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.ou_noise = OUNoise(action_size, random_seed)
        self.uni_noise = UniformNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, t_step, num_agents):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory and UPDATE_STEP steps have passed
        if (len(self.memory) > BATCH_SIZE) and ((t_step % UPDATE_STEP) == 0):
            for _ in range(UPDATE_ITER):
                experiences = self.memory.sample()                 # Sample experience from memory
                for agent_id in range(num_agents):                 # Learn from all agents
                    self.learn(experiences, GAMMA, agent_id)


    def act(self, state, noise_prob = 1, noise_scale = 1, noise_type=None, noise_apply=None):
        """Returns actions for given state as per current policy.
        Params
        ======
            noise_prob (float): probability to add noise
            noise_scale (float): scaling factor for noise
            noise_type (str): what type of noise should be added
            noise_apply (str): how the noise should be applied
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
            
        if noise_apply == 'fuse':
            if noise_type == 'ou':
                action = (1-noise_prob) * action + (noise_prob) * noise_scale * self.ou_noise.sample()
            elif noise_type == 'uni':
                action = (1-noise_prob) * action + (noise_prob) * noise_scale * self.uni_noise.sample()
        elif noise_apply == 'thres':
            if noise_type == 'ou' and noise_prob > random.random():
                action = noise_scale*self.ou_noise.sample()
            elif noise_type == 'uni' and noise_prob > random.random():
                action = noise_scale*self.uni_noise.sample()
        elif noise_apply == 'add':
            if noise_type == 'ou':
                action += noise_scale*noise_prob * self.ou_noise.sample()
            elif noise_type == 'uni' and noise_prob > random.random():
                action += noise_scale*noise_prob * self.uni_noise.sample()        
        elif noise_apply == None:
            pass
        else:
            print('Invalid Noise Type!')

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
      
        #Actions from all agents
        actions_next = actions.clone()
        #Inject actor network prediction for active agent
        actions_next[:,agent_id*self.action_size:(agent_id+1)*self.action_size] = self.actor_target(next_states)
        
        # Get predicted next-state actions and Q values from target models  
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
      
        # ---------------------------- update actor ---------------------------- #
        
        # Actions from all agents
        actions_pred = actions.clone()
        # Inject actor network prediction for active agent
        actions_pred[:,agent_id*self.action_size:(agent_id+1)*self.action_size] = self.actor_local(states)
                
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def load(self, actor_file, critic_file):
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.actor_target.load_state_dict(torch.load(actor_file))
        self.critic_local.load_state_dict(torch.load(critic_file))
        self.critic_target.load_state_dict(torch.load(critic_file))

        
import numpy as np
np.random.seed(0)
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal noise to mean (mu)."""
        self.noise = copy.copy(self.mu)

    def sample(self):
        """Update internal noise and return the current sample."""
        dx = self.theta * (self.mu - self.noise) + self.sigma * np.random.randn(self.size)
        self.noise = self.noise + dx
        return self.noise
        
        

class UniformNoise:
    """Uniform noise."""

    def __init__(self, size, seed, mu=0., lower=-1, upper=1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.lower = lower
        self.upper = upper
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        pass

    def sample(self):
        """Update internal noise and return the current sample."""
        self.noise = np.random.uniform(self.lower, self.upper, self.size) + self.mu
        return self.noise
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)