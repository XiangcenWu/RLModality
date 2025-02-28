import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Training.data_loading import readh5
from Training.training_helpers import post_process, dice_coefficient
import torch



class Env():

    def __init__(self, patient: str, seg_model):
        patient = readh5(patient)

        self.t2 = patient['t2']
        self.hb = patient['hb']
        self.gt = patient['gt']
        self.shape = self.hb.shape
        
        self.both_seg = self.get_inference_output(seg_model, torch.cat([self.t2, self.hb]))
        self.t2_seg = self.get_inference_output(seg_model, torch.cat([self.t2, torch.zeros_like(self.hb)]))
        self.hb_seg = self.get_inference_output(seg_model, torch.cat([torch.zeros_like(self.t2), self.hb]))
        

        self.current_segmentation = torch.zeros(size=self.shape)
        self.last_action = torch.zeros(self.shape)
        
        
        self.mean_dice = self.get_mean_accuracy()
        self.worst_dice = self.get_worse_accuracy()
        self.best_dice = self.get_best_accuracy()
        


        
        
        self.index_128 = [slice(0, 43), slice(43, 86), slice(86, 128)]
        self.index_32 = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20), slice(20, 24), slice(24, 28), slice(28, 32)]
        # self.index_32 = [slice(0, 8), slice(8, 16), slice(16, 24), slice(24, 32)]
        
        self.current_accuracy = 0.
        self.last_current_accuracy = 0.
##########################
    @property
    def all_zero(self):
        return all(num == 0 for num in (self.mean_dice, self.worst_dice, self.best_dice))

    def get_inference_output(self, seg_model, input_tensor):
        with torch.no_grad():
            output = seg_model(input_tensor.unsqueeze(0))
            output = post_process(output).squeeze(0)
        return output
###########################


    def reset(self):
        return torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ])
    
    

    def step_train(self, action):
        self.update_current_seg(action)
        self.current_accuracy = self.calculate_current_accuracy()

        reward = self.current_accuracy - self.last_current_accuracy
        self.last_current_accuracy = self.current_accuracy
        
        
        # if reward > 0:
        #     reward = 1.
        # else:
        #     reward = 0.
        

        obs = torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ])
        

        return obs, reward


    def update_current_seg(self, action):


        if action <= 7: # both
            d = self.index_32[action]
            self.current_segmentation[:, :, :, d] = self.both_seg[:, :, :, d]
        elif action > 7 and action <= 15: # t2
            d = self.index_32[action - 8]
            self.current_segmentation[:, :, :, d] = self.t2_seg[:, :, :, d]
        elif action > 15 and action <= 23: # hb
            d = self.index_32[action - 16]
            self.current_segmentation[:, :, :, d] = self.hb_seg[:, :, :, d]
        else: # nothing
            d = self.index_32[action - 24]
            self.current_segmentation[:, :, :, d] = 0.



    def calculate_current_accuracy(self): #
        # post set to false because all inference output has been post process according to self.get_inference_output
        return dice_coefficient(self.current_segmentation, self.gt, post=False).item()
    
    
    def get_all_accuracy(self):

        
        both = dice_coefficient(self.both_seg, self.gt, post=False).item()
        t2 = dice_coefficient(self.t2_seg, self.gt, post=False).item()
        hb = dice_coefficient(self.hb_seg, self.gt, post=False).item()
        
        
        return t2, hb, both
    
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
        
        if random.random() < 0.5:
            self.current_segmentation = random.choice([self.both_seg, self.t2_seg, self.hb_seg])
        
        for _ in range(loop_per_patient):
            action = (
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
        
        
        
        
    def get_both_state_accuracy(self):
        
        return torch.cat([
            self.t2,
            self.hb,
            self.both_seg,
        ]), dice_coefficient(self.both_seg, self.gt, post=False).item()


    def get_t2_state_accuracy(self):
        
        return torch.cat([
            self.t2,
            self.hb,
            self.t2_seg,
        ]), dice_coefficient(self.t2_seg, self.gt, post=False).item()
        
    def get_hb_state_accuracy(self):
        
        return torch.cat([
            self.t2,
            self.hb,
            self.hb_seg,
        ]), dice_coefficient(self.hb_seg, self.gt, post=False).item()
        
    def get_null_state_accuracy(self):

        
        return torch.cat([
            self.t2,
            self.hb,
            torch.zeros_like(self.t2)
        ]), 0

