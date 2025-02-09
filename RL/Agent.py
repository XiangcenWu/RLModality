import torch
import torch.nn as nn
from monai.networks.nets import ViT, DenseNet
import os
from torch.distributions.categorical import Categorical
import random
from RL.vit_classifier import vit_classifier

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


        return torch.stack(self.states),\
                torch.stack(actions),\
                torch.tensor(self.rewards),\
                batches

    def store_memory(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []


    
    
class Agent:
    def __init__(self, gamma=0.95, batch_size=64, n_epochs=10, device='cpu'):
        self.gamma = gamma

        self.n_epochs = n_epochs
        
        self.device = device

        self.actor = vit_classifier().to(device)
        self.memory = PPOMemory(batch_size)
        
        
        # torch.nn.init.kaiming_uniform_(self.actor.weight, nonlinearity='relu')


    def remember(self, state, action, reward):
        self.memory.store_memory(state, action,reward)


    def choose_action(self, state, noise=0.3):
        
        self.actor.eval()
        
        
        with torch.no_grad():
            features = self.actor(state)
            features = torch.softmax(features, dim=-1)


        if random.random() < noise:
            action = torch.randint(0, 32, size=(1, 1)).item()
        else:
            action = Categorical(features).sample().item()

        return action, features.squeeze(0)
    
    def calculate_cdr(self, reward_arr, ):

        cdr = torch.zeros(len(reward_arr), dtype=float, device=self.device)
        for t in range(len(reward_arr)):
            discount = 1.
            cdr_t = 0.
            for k in range(t, len(reward_arr)):
                cdr_t += discount * reward_arr[k]
                discount *= self.gamma
            cdr[t] = cdr_t
            
        # cdr = (cdr - cdr.min()) / (cdr.max() - cdr.min())
            
        return cdr


    def learn_reinforce(self, optimizer):
        self.actor.train()

        # epoch loop
        for _ in range(self.n_epochs):
            state_arr, action_arr, reward_arr, batches = \
                    self.memory.generate_batches()
            cdr_arr = self.calculate_cdr(reward_arr)

            action_arr = action_arr.to(self.device)
            



            state_arr = state_arr.to(self.device)
            cdr_arr = cdr_arr.to(self.device)
            
            
            
            # print(state_arr.shape, action_arr, reward_arr, cdr_arr, batches)


            for batch in batches:
                states = state_arr[batch]
                actions = action_arr[batch]
                cdr = cdr_arr[batch]
                
                # print(actions)
                raw_output = self.actor(states)
                # print(raw_output)
                probs = torch.softmax(raw_output, dim=-1)
                print(probs)
                print(actions)
                print(cdr)
                
                dist = Categorical(probs)
                # print(f'this is dist {dist}')

                
                log_probs = dist.log_prob(actions)
                
                loss = - cdr * log_probs
                
                
                
                
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()

                
                

            


        self.memory.clear_memory()
        
        
        
    def save_models(self, actor_file_name):
        torch.save(self.actor.state_dict(), actor_file_name)
        print('actor saved!')
        
        
    def load_models(self, actor_file_name):
        self.actor.load_state_dict(torch.load(actor_file_name, map_location='cpu', weights_only=True))
        print('actor loaded!')



