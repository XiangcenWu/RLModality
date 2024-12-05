import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Training.data_loading import readh5
from Training.training_helpers import post_process, dice_coefficient
import torch



class Env():

    def __init__(self, patient: str, seg_model, eps_length):
        patient = readh5(patient)

        self.t2 = patient['t2']
        self.hb = patient['hb']
        self.gt = patient['gt']
        self.shape = self.hb.shape
        
        self.both_seg = self.get_inference_output(seg_model, torch.cat([self.t2, self.hb]))
        self.t2_seg = self.get_inference_output(seg_model, torch.cat([self.t2, torch.zeros_like(self.hb)]))
        self.hb_seg = self.get_inference_output(seg_model, torch.cat([torch.zeros_like(self.t2), self.hb]))
        

        self.current_segmentation = torch.zeros(size=self.shape)
        self.last_action = None
        
        
        self.mean_dice = self.get_mean_accuracy()
        self.worst_dice = self.get_worse_accuracy()
        self.best_dice = self.get_best_accuracy()
        
        self.max_length = eps_length
        self.current_length = 0
        
        
        self.index_128 = [slice(0, 43), slice(43, 86), slice(86, 128)]
        self.index_32 = [slice(0, 8), slice(8, 16), slice(16, 24), slice(24, 32)]


    def get_inference_output(self, seg_model, input_tensor):
        with torch.no_grad():
            output = seg_model(input_tensor.unsqueeze(0))
            output = post_process(output).squeeze(0)
        return output



    def reset(self):
        self.current_length += 1
        return torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
            # torch.zeros(size=self.shape)
        ])
    
    

    def step_train(self, action):
        self.update_current_seg(action)
        # reward = 0.
        
        self.current_accuracy = self.calculate_current_accuracy()
        reward = self.current_accuracy - self.mean_dice
        # if self.current_accuracy > self.best_dice:
        #     reward = 2


        if self.current_length == self.max_length:
            self.current_accuracy = self.calculate_current_accuracy()
            if self.current_accuracy > self.best_dice:
                reward = 10.
            elif self.current_accuracy > self.mean_dice:
                reward = 5.
            # elif self.current_accuracy > self.worst_dice:
            #     reward = 2.5
            else:
                reward = -10
        self.current_length += 1


        obs = torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
            # self.get_current_location(action)
        ])

        return obs, reward
    
    
    def update_current_seg(self, action):
        width, heigth, depth, modality = action
        w = self.index_128[width]
        h = self.index_128[heigth]
        d = self.index_32[depth]
        if modality == 0: # both
            self.current_segmentation[:, w, h, d] = self.both_seg[:, w, h, d]
        elif modality == 1: # t2
            self.current_segmentation[:, w, h, d] = self.t2_seg[:, w, h, d]
        elif modality == 2: # hb
            self.current_segmentation[:, w, h, d] = self.hb_seg[:, w, h, d]
        else: # nothing
            self.current_segmentation[:, w, h, d] = 0.


    def calculate_current_accuracy(self): #
        # post set to false because all inference output has been post process according to self.get_inference_output
        return dice_coefficient(self.current_segmentation, self.gt, post=False).item()
    
    
    def get_all_accuracy(self):

        
        both = dice_coefficient(self.both_seg, self.gt, post=False).item()
        t2 = dice_coefficient(self.t2_seg, self.gt, post=False).item()
        hb = dice_coefficient(self.hb_seg, self.gt, post=False).item()
        
        
        return both, t2, hb
    
    def get_best_accuracy(self):
        both, t2, hb = self.get_all_accuracy()
        return torch.tensor([both, t2, hb]).max().item()
    
    def get_worse_accuracy(self):
        both, t2, hb = self.get_all_accuracy()
        return torch.tensor([both, t2, hb]).min().item()
    
    def get_mean_accuracy(self):
        both, t2, hb = self.get_all_accuracy()
        return torch.tensor([both, t2, hb]).mean().item()
    
    
    def get_rand_state_accuracy(self, loop_per_patient):
        
        for _ in range(loop_per_patient):
            action = (
                        torch.randint(0, 3, size=(1, )).item(),
                        torch.randint(0, 3, size=(1, )).item(),
                        torch.randint(0, 4, size=(1, )).item(),
                        torch.randint(0, 4, size=(1, )).item()
                    )
            self.update_current_seg(action)
        accuracy = self.calculate_current_accuracy()


        return torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ]), accuracy
        
    def clear_current_segmentation(self):
        self.current_segmentation = torch.zeros(size=self.shape)


