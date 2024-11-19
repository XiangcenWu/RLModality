import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Training.data_loading import readh5
from Training.training_helpers import post_process
import torch



class Env():
    
    
    
    def __init__(self, patient: str, seg_model, weak_model):
        patient = readh5(patient)
        
        self.t2 = patient['t2']
        self.hb = patient['hb']
        self.shape = self.hb.shape
        
        self.both_seg = self.get_inference_output(seg_model, torch.cat([self.t2, self.hb]))
        self.t2_seg = self.get_inference_output(seg_model, torch.cat([self.t2, torch.zeros_like(self.hb)]))
        self.hb_seg = self.get_inference_output(seg_model, torch.cat([torch.zeros_like(self.t2), self.hb]))
        
        
        
        self.weak_model = weak_model
        
        
        
        self.current_index = 15
        self.current_segmentation = torch.zeros(size=self.shape)
        self.current_weak_accuracy = 0


    def get_inference_output(self, seg_model, input_tensor):
        with torch.no_grad():
            output = seg_model(input_tensor.unsqueeze(0))
            # output = post_process(output).squeeze(0)
    
        return output

    def generate_location_volumn(self, index):
        lv = torch.zeros(size=self.shape)
        lv[..., index] = 1.
        return lv
    
    
    def reset(self):
        return torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
            self.generate_location_volumn(self.current_index)
        ])
    
    
    def step(self, action):
        reward = self.calculate_reward(action)

        obs = torch.cat([
            self.t2,
            self.hb,
            self.current_segmentation,
            self.generate_location_volumn(self.current_index)
        ])

        return obs, reward
    
    ###########################
    ###########################
    # when inferencing weak net, you should not post process the segmentation predictions
    def calculate_reward(self, action): #
        slice_action, modality_action = action
        if self.current_index == 0 and slice_action == 0:
            index_now = 31
        elif self.current_index == 31 and slice_action == 1:
            index_now = 0
        elif slice_action == 0:
            index_now = self.current_index - 1
        elif slice_action == 1:
            index_now = self.current_index + 1


        if modality_action == 0:
            self.current_segmentation[..., index_now] = self.both_seg[..., index_now]
        elif modality_action == 1:
            self.current_segmentation[..., index_now] = self.t2_seg[..., index_now]
        elif modality_action == 2:
            self.current_segmentation[..., index_now] = self.hb_seg[..., index_now]
        elif modality_action == 3:
            self.current_segmentation[..., index_now] = 0  ### convert segmentation
        self.current_index = index_now ###### convert indexx
        
        last_weak_accuracy = self.current_weak_accuracy
        with torch.no_grad():
            weak_intput = torch.cat([self.t2, self.hb, self.current_segmentation]).unsqueeze(0)
            self.current_weak_accuracy = self.weak_model(weak_intput).item() ##### convert weak accuracy
            print('current weak accuracy', self.current_weak_accuracy)
        reward = self.current_weak_accuracy - last_weak_accuracy
            
        

        
        return reward
    
    
if __name__ == "__main__":
    env = Env("/home/xiangcen/RLModality/picai_h5/39.h5", None, None)