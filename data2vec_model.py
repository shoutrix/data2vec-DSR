import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
import copy


class D2R(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.config = config
        # self.model_name = config.model_name
        
        model_name="facebook/wav2vec2-large-lv60"
        self.teacher = Wav2Vec2Model.from_pretrained(model_name)        
        self.student = Wav2Vec2Model.from_pretrained(model_name)
        self.teacher.feature_extractor._freeze_parameters()
        self.student.feature_extractor._freeze_parameters()
        self.teacher.requires_grad_(False)
        self.MSE_loss_layers = [12, 13, 14]

        self.MSEloss = nn.MSELoss()

    def forward(self, student_input, teacher_input, alpha):
        
        with torch.no_grad():
            # student feature extraction
            student_features = self.student.feature_extractor(student_input)
            student_features = student_features.transpose(1, 2)
            student_hidden_states, student_features = self.student.feature_projection(student_features)
            
            # teacher feature extraction
            teacher_features = self.teacher.feature_extractor(teacher_input)
            teacher_features = teacher_features.transpose(1, 2)
            teacher_hidden_states, teacher_features = self.teacher.feature_projection(teacher_features)

            # teacher encoder forward
            teacher_encoder_out = self.teacher.encoder(teacher_hidden_states, output_hidden_states=True)
            teacher_last_hidden_state = teacher_encoder_out.last_hidden_state
            teacher_hidden_states = teacher_encoder_out.hidden_states
            
        # student feature masking
        student_hidden_states, mask = self.mask_student_hidden_states(student_hidden_states)
        # mask : shape(B, T) True where masked
            
        # student encoder forward
        student_encoder_out = self.student.encoder(student_hidden_states, output_hidden_states=True)
        student_last_hidden_state = student_encoder_out.last_hidden_state
        student_hidden_states = student_encoder_out.hidden_states
        
        
        print(len(student_hidden_states))
        print(student_hidden_states[-1])
        print(student_last_hidden_state)
        
        # MSE loss over several layers of teacher and student
        layer_loss, mean_loss = self.compute_mse_loss(teacher_hidden_states, student_hidden_states, mask)

        # compute total loss
        return mean_loss
    
    
    @classmethod
    def align_with_dtw(cls, teacher_features, student_features):
        B, T1, d = teacher_features.shape
        _, T2, _ = student_features.shape

        if T1 == T2:
            return teacher_features

        expanded_teacher_features = torch.zeros_like(student_features)

        for b in range(B):
            _, path = fastdtw(teacher_features[b].detach().cpu().numpy(), student_features[b].detach().cpu().numpy(), dist=euclidean)
            
            # print(path)

            teacher_indices, student_indices = zip(*path)
            teacher_indices = torch.tensor(teacher_indices, dtype=torch.long, device=teacher_features.device)
            student_indices = torch.tensor(student_indices, dtype=torch.long, device=teacher_features.device)
            expanded_teacher_features[b] = expanded_teacher_features[b] + teacher_features[b, teacher_indices]
        return expanded_teacher_features

        

    def mask_student_hidden_states(self, student_hidden_states):
        B, T, _ = student_hidden_states.shape
        mask = torch.ones((B, T), dtype=torch.bool)
        return student_hidden_states, mask
    
    
    def compute_mse_loss(self, teacher_hidden_states, student_hidden_states, mask):
        
        dim_ = teacher_hidden_states[0].shape[-1]
        layer_loss = {}
        
        for layer_idx in self.MSE_loss_layers:
            teacher_hidden_state_layer = teacher_hidden_states[layer_idx]
            student_hidden_state_layer = student_hidden_states[layer_idx]
            teacher_hidden_state_layer = self.align_with_dtw(teacher_hidden_state_layer, student_hidden_state_layer)
            teacher_hidden_state_layer = teacher_hidden_state_layer[mask].view(-1, dim_)
            student_hidden_state_layer = student_hidden_state_layer[mask].view(-1, dim_)
            
            loss = F.mse_loss(teacher_hidden_state_layer, student_hidden_state_layer, reduce="mean")
            layer_loss[layer_idx] = loss
            
            print(loss)
            
        mean_loss = torch.tensor(list(layer_loss.values())).mean()
        return layer_loss, mean_loss
    
    
    def compute_cc_loss(self, teacher_last_hidden_state, student_last_hidden_state, masked_indices):
        return None
    
  
  
def main():
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = D2R().to(device)
    optimizer = torch.optim.AdamW(model.teacher.parameters(), lr=1e-5)

    # Dummy tensors to simulate audio input
    student_audio = torch.randn(1, 48000).to(device)  # Batch of 2, 1 sec of audio (16kHz)
    teacher_audio = torch.randn(1, 16000).to(device)  # Corresponding regular speech

    # Training step
    model.train()
    optimizer.zero_grad()
    loss = model(student_audio, teacher_audio, 0.3)
    # loss.backward()
    # optimizer.step()

    # print(f"Training Loss: {loss.item()}")

if __name__ == "__main__":
    main()