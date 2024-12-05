import torch
import torch.nn as nn
from monai.networks.nets import ViT, DenseNet
import os
from torch.distributions.categorical import Categorical
class PPOMemory:
    def __init__(self, batch_size):
        self.states = [] # store 4, 128, 128, 32 tensor
        self.probs = [] # store tuple
        self.vals = [] # store int
        self.actions = [] # store tuple
        self.rewards = [] # store int

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.randperm(n_states, dtype=int)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        actions = [torch.tensor(item) for item in self.actions]
        probs = [torch.tensor(item) for item in self.probs]

        return torch.stack(self.states),\
                torch.stack(actions),\
                torch.stack(probs),\
                torch.tensor(self.vals),\
                torch.tensor(self.rewards),\
                batches

    def store_memory(self, state, action, probs, vals, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)


    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []



class ActorModelViT(nn.Module):
    
    def __init__(self, in_channels=3, img_size=(128, 128, 32), patch_size=(16, 16, 8), hidden_size=512):
        super().__init__()
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_layers=24,
            classification=True,
            post_activation="Tanh",
            num_classes=hidden_size,
            dropout_rate = 0,
        )
        self.vit_vector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.linear_h = nn.Linear(hidden_size, 3)
        self.linear_w = nn.Linear(hidden_size, 3)
        self.linear_d = nn.Linear(hidden_size, 4)
        self.linear_modality = nn.Linear(hidden_size, 4)
        
    def forward(self, x, noise=10):
        feature = self.vit(x)[0]
        feature = self.vit_vector(feature)
        

        dist_h = self.linear_h(feature)
        dist_w = self.linear_w(feature)
        dist_d = self.linear_d(feature)
        dist_modality = self.linear_modality(feature)
        
        

        
        
        dist_h = self.linear_h(feature)
        dist_w = self.linear_w(feature)
        dist_d = self.linear_d(feature)
        dist_modality = self.linear_modality(feature)
        
        # if noise:
        #     dist_h += noise*torch.rand_like(dist_h, device=dist_h.device)
        #     dist_w += noise*torch.rand_like(dist_w, device=dist_w.device)
        #     dist_d += noise*torch.rand_like(dist_d, device=dist_d.device)
        #     dist_modality += noise*torch.rand_like(dist_modality, device=dist_modality.device)
        if noise:
            # Using Gaussian noise instead of uniform noise for smoother exploration
            noise_h = torch.normal(mean=0, std=noise, size=dist_h.shape).to(dist_h.device)
            noise_w = torch.normal(mean=0, std=noise, size=dist_w.shape).to(dist_w.device)
            noise_d = torch.normal(mean=0, std=noise, size=dist_d.shape).to(dist_d.device)
            noise_modality = torch.normal(mean=0, std=noise, size=dist_modality.shape).to(dist_modality.device)

            # Add noise to each of the distributions
            dist_h += noise_h
            dist_w += noise_w
            dist_d += noise_d
            dist_modality += noise_modality
            
        dist_h = torch.softmax(dist_h, dim=-1)
        dist_w = torch.softmax(dist_w, dim=-1)
        dist_d = torch.softmax(dist_d, dim=-1)
        dist_modality = torch.softmax(dist_modality, dim=-1)
            


        return Categorical(dist_h), \
                Categorical(dist_w), \
                Categorical(dist_d), \
                Categorical(dist_modality), 

            
class CriticModelViT(nn.Module):
    
    def __init__(self, in_channels=3, img_size=(128, 128, 32), patch_size=(16, 16, 8)):
        super().__init__()
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_layers=16,
            classification=True,
            post_activation=None,
            num_classes=1,
            qkv_bias = True,
            save_attn = True
        )
        self.rand_model = DenseNet(3, 3, 1)
        # self.rand_model = nn.Sequential(
        #     # Input shape: (batch, 4, 128, 128, 32)
        #     nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),  # (batch, 16, 128, 128, 32)
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample: (batch, 16, 64, 64, 16)
            
        #     nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (batch, 32, 64, 64, 16)
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample: (batch, 32, 32, 32, 8)
            
        #     nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (batch, 64, 32, 32, 8)
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample: (batch, 64, 16, 16, 4)
            
        #     nn.Flatten(),  # Flatten to (batch, 64 * 16 * 16 * 4)
            
        #     nn.Linear(64 * 16 * 16 * 4, 128),  # Fully connected layer
        #     nn.ReLU(),
        #     nn.Linear(128, 1)  # Output a single number
        # )
        
    def forward(self, x):
        # feature = self.vit(x)[0]
        feature = self.rand_model(x)

        return feature


class ActorNetwork(nn.Module):
    def __init__(self, alpha=0.0001, device='cpu', chkpt_dir='models/actor'):
        super(ActorNetwork, self).__init__()

        # self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.ptm')
        self.model = ActorModelViT()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state, noise=False):
        return self.model(state, noise=noise)


class CriticNetwork(nn.Module):
    def __init__(self, alpha=0.0001, device='cpu'):
        super().__init__()
        self.model = CriticModelViT()
        # self.model = DenseNet(3, 4, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        value = self.model(state)
        return value
    
    
    
    
class Agent:
    def __init__(self, gamma=0.95, alpha=0.0003, gae_lambda=0.9,
            policy_clip=0.2, batch_size=64, n_epochs=10, device='cpu'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.device = device

        self.actor = ActorNetwork(alpha=alpha).to(device)
        self.critic = CriticNetwork(alpha=alpha).to(device)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward):
        self.memory.store_memory(state, action, probs, vals, reward)


    def choose_action(self, state, noise=10):
        
        self.actor.eval()
        self.critic.eval()


        dist_h, dist_w, dist_d, dist_modality = self.actor(state, noise=noise)
        value = self.critic(state)
        action_h, action_w, action_d, action_modality = dist_h.sample(), dist_w.sample(), dist_d.sample(), dist_modality.sample()


        probs_h, probs_w, probs_d, probs_modality = torch.squeeze(dist_h.log_prob(action_h)).item(), \
                                                    torch.squeeze(dist_w.log_prob(action_w)).item(), \
                                                    torch.squeeze(dist_d.log_prob(action_d)).item(), \
                                                    torch.squeeze(dist_modality.log_prob(action_modality)).item(), 
        action_h, action_w, action_d, action_modality = torch.squeeze(action_h).item(), \
                                                        torch.squeeze(action_w).item(), \
                                                        torch.squeeze(action_d).item(), \
                                                        torch.squeeze(action_modality).item()
        value = torch.squeeze(value).item()

        return (action_h, action_w, action_d, action_modality), \
               (probs_h, probs_w, probs_d, probs_modality), \
                value
                


    def calculate_adv(self, reward_arr, value_arr):
        print('reward:,', reward_arr, '\n', 'val:', value_arr)
        advantage = torch.zeros(len(reward_arr), dtype=float, device=self.device)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*value_arr[k+1] - value_arr[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        print('adv: ', advantage)

        
        return advantage
    
    def calculate_cdr(self, reward_arr, ):

        cdr = torch.zeros(len(reward_arr), dtype=float, device=self.device)
        for t in range(len(reward_arr)):
            discount = 1.
            cdr_t = 0.
            for k in range(t, len(reward_arr)):
                cdr_t += discount * reward_arr[k]
                discount *= self.gamma
            cdr[t] = cdr_t
            
        print('cdr:', cdr)
        return cdr


    def learn(self):
        self.actor.train()
        self.critic.train()
        # epoch loop
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, batches = \
                    self.memory.generate_batches()
            vals_arr = vals_arr.to(self.device)
            advantage = self.calculate_adv(reward_arr, vals_arr)
            cdr = self.calculate_cdr(reward_arr=reward_arr)
            




            # bathc loop
            for batch in batches:
                states = state_arr[batch].to(self.device)
                old_probs = old_prob_arr[batch].to(self.device)
                old_probs_h, old_probs_w, old_probs_d, old_probs_modality = torch.chunk(old_probs, chunks=4, dim=1)

                ##
                actions = action_arr[batch].to(self.device)
                action_h, action_w, action_d, action_modality = torch.chunk(actions, chunks=4, dim=1)


                dist_h, dist_w, dist_d, dist_modality = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs_h, new_probs_w, new_probs_d, new_probs_modality = \
                    dist_h.log_prob(action_h), dist_w.log_prob(action_w), dist_d.log_prob(action_d), dist_modality.log_prob(action_modality)
                

                
                prob_ratio = \
                    (new_probs_h.exp() / old_probs_h.exp()) + (new_probs_w.exp() / old_probs_w.exp()) + \
                    (new_probs_d.exp() / old_probs_d.exp()) + (new_probs_modality.exp() / old_probs_modality.exp())
                # prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = \
                #     (new_probs_h - old_probs_h).exp() + (new_probs_w - old_probs_w).exp() + \
                #     (new_probs_d - old_probs_d).exp() + (new_probs_modality - old_probs_modality).exp()
                prob_ratio = prob_ratio.mean()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                #############################################################
                
                # returns = advantage[batch] #+ vals_arr[batch]

                critic_loss = (cdr[batch]  - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                # total_loss = 0.1*critic_loss


                ##############################################################
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        
        