import torch
import torch.nn as nn
from monai.networks.nets import ViT, DenseNet
import os
from torch.distributions.categorical import Categorical
import random


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


class ActorModelDense(nn.Module):
    
    def __init__(self, in_channels=3, hidden_size=1536):
        super().__init__()
        self.dense = DenseNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=32,
            init_features=128,
            growth_rate=64,
            block_config= (14, 26, 50, 34)
        )
        # self.dense_vector = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )


        # self.linear_modality = nn.Sequential(
        #     nn.Linear(hidden_size, 600),
        #     nn.ReLU(),
        #     nn.Linear(600, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 32)
        # )
        
    def forward(self, x):
        feature = self.dense(x)
        # feature = self.dense_vector(feature)


        # dist_modality = self.linear_modality(feature)


        # dist_modality = self.linear_modality(feature)


        # dist_modality = torch.sigmoid(dist_modality)

        # print(dist_modality)

        feature = torch.sigmoid(feature)
        print(feature)
        return Categorical(feature)
    

class CriticModelDense(nn.Module):
    
    def __init__(self, in_channels=3):
        super().__init__()
        self.dense = DenseNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            
            init_features=128,
            growth_rate=64,
            block_config= (12, 24, 48, 32)
        )
    def forward(self, x):
        feature = self.dense(x)
        return feature

class ActorNetwork(nn.Module):
    def __init__(self, alpha=0.0001):
        super().__init__()

        self.model = ActorModelDense()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)


    def forward(self, state):
        return self.model(state)


class CriticNetwork(nn.Module):
    def __init__(self, alpha=0.0001):
        super().__init__()
        self.model = CriticModelDense()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)


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


    def choose_action(self, state, noise=0.5):
        
        self.actor.eval()
        self.critic.eval()


        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic(state)

        if random.random() < noise:
            action = torch.randint(0, 32, size=(1, 1)).to(self.device)
         
        else:
            action = dist.sample()

        
        probs = torch.squeeze(dist.log_prob(action)).item()
        
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, \
               probs, \
                value



    def calculate_adv(self, reward_arr, value_arr):

        advantage = torch.zeros(len(reward_arr), dtype=float, device=self.device)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*value_arr[k+1] - value_arr[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t

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
            
        return cdr


    def learn(self):
        self.actor.train()
        self.critic.train()
        # epoch loop
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, batches = \
                    self.memory.generate_batches()
            print(state_arr.shape, action_arr.shape, old_prob_arr.shape, vals_arr.shape,\
            reward_arr.shape, batches)
            vals_arr = vals_arr.to(self.device)
            advantage = self.calculate_adv(reward_arr, vals_arr)
            # cdr = self.calculate_cdr(reward_arr=reward_arr)





            # bathc loop
            for batch in batches:
                states = state_arr[batch].to(self.device)
                old_probs = old_prob_arr[batch].to(self.device)

                

                ##
                actions = action_arr[batch].to(self.device).flatten()
                

                
                dist= self.actor(states)

                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)




                prob_ratio = (new_probs - old_probs).exp()


                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                #############################################################
                


                critic_loss = (advantage[batch] + vals_arr[batch] - critic_value[batch]) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss


                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        
        
        
    def save_models(self, actor_file_name, critic_file_name):
        torch.save(self.actor.state_dict(), actor_file_name)
        torch.save(self.critic.state_dict(), critic_file_name)
        print('actor and critic saved!')
        
        
    def load_models(self, actor_file_name, critic_file_name):
        self.actor.load_state_dict(torch.load(actor_file_name, map_location='cpu', weights_only=True))
        self.critic.load_state_dict(torch.load(critic_file_name, map_location='cpu', weights_only=True))
        print('actor and critic loaded!')



