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
        self.current_accuracy = 0.
        
        
        self.index_128 = [slice(0, 43), slice(43, 86), slice(86, 128)]
        self.index_32 = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20), slice(20, 24), slice(24, 28), slice(28, 32)]


    def get_inference_output(self, seg_model, input_tensor):
        with torch.no_grad():
            output = seg_model(input_tensor.unsqueeze(0))
            output = post_process(output).squeeze(0)
        return output



    def reset(self):
        return torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ])
    
    
    def step_inference(self, action):
        self.update_current_seg(action)
        reward = self.calculate_current_accuracy()

        obs = torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ])

        return obs, reward
    
    def step_train(self, action):
        last_accuracy = self.current_accuracy
        self.update_current_seg(action)
        
        self.current_accuracy = self.calculate_current_accuracy()
        reward = self.current_accuracy - last_accuracy

        obs = torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
        ])
        if reward < 0:
            reward *= 5
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
        return dice_coefficient(self.current_segmentation, self.gt, post=False)
    
    
    def get_all_accuracy(self):
        # self.both_seg = self.get_inference_output(seg_model, torch.cat([self.t2, self.hb]))
        # self.t2_seg = self.get_inference_output(seg_model, torch.cat([self.t2, torch.zeros_like(self.hb)]))
        # self.hb_seg = self.get_inference_output(seg_model, torch.cat([torch.zeros_like(self.t2), self.hb]))
        
        
        both = dice_coefficient(self.both_seg, self.gt, post=False).item()
        t2 = dice_coefficient(self.t2_seg, self.gt, post=False).item()
        hb = dice_coefficient(self.hb_seg, self.gt, post=False).item()
        
        
        return both, t2, hb
        
    
    
