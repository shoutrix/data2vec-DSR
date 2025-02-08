import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


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
            # teacher feature extraction
            teacher_features = self.teacher.feature_extractor(teacher_input)
            teacher_features = teacher_features.transpose(1, 2)
            teacher_hidden_states, teacher_features = self.teacher.feature_projection(teacher_features)
            
            # student feature extraction
            student_features = self.student.feature_extractor(student_input)
            student_features = student_features.transpose(1, 2)
            student_hidden_states, student_features = self.student.feature_projection(student_features)
            
            # DTW and teacher feature repeatation
            print(teacher_hidden_states.shape, student_hidden_states.shape)
            teacher_hidden_states, student_hidden_states = self.align_with_dtw(teacher_hidden_states, student_hidden_states)
            print(teacher_hidden_states.shape)
            
            # student feature masking
            # student_hidden_states, masked_indices = self.mask_student_hidden_states(student_hidden_states)

            # teacher encoder forward
            # teacher_encoder_out = self.teacher.encoder(teacher_hidden_states, output_hidden_states=True)
            # teacher_last_hidden_state = teacher_encoder_out.last_hidden_state
            # teacher_hidden_states = teacher_encoder_out.hidden_states
            
            
        # student encoder forward
        # student_encoder_out = self.student.encoder(student_hidden_states, output_hidden_states=True)
        # student_last_hidden_state = student_encoder_out.last_hidden_state
        # student_hidden_states = student_encoder_out.hidden_states
        
        # MSE loss over several layers of teacher and student
        # mse_loss = self.compute_mse_loss(teacher_hidden_states, student_hidden_states)
        
        # (optional) compute Cross-Contrastive-Loss
        # cc_loss = self.compute_cc_loss(teacher_last_hidden_state, student_last_hidden_state, masked_indices)
        
        # compute total loss
        # loss = alpha * mse_loss + (1-alpha) * cc_loss
        # return loss

    @classmethod
    def align_with_dtw(cls, teacher_features, student_features):

        B, T1, d = teacher_features.shape
        _, T2, _ = student_features.shape

        if T1 == T2:
            return teacher_features, student_features
        if T1 < T2:
            small_seq, large_seq = teacher_features, student_features
        else:
            small_seq, large_seq = student_features, teacher_features

        _, T_small, _ = small_seq.shape
        _, T_large, _ = large_seq.shape

        expanded_small_seq = torch.zeros((B, T_large, d), dtype=small_seq.dtype, device=small_seq.device)

        for b in range(B):
            _, path = fastdtw(small_seq[b].cpu().numpy(), large_seq[b].cpu().numpy(), dist=euclidean)
            print(path)
            small_indices, large_indices = zip(*path)

            expanded_seq = torch.zeros((T_large, d), dtype=small_seq.dtype, device=small_seq.device)
            for xi, yi in zip(small_indices, large_indices):
                expanded_seq[yi] += small_seq[b, xi]

            expanded_small_seq[b] = expanded_seq

        if T1 < T2:
            return expanded_small_seq, student_features
        else:
            return teacher_features, expanded_small_seq

        

    def mask_student_hidden_states(self):
        return None
    
    
    def compute_mse_loss(self, teacher_hidden_states, student_hidden_states):
        return None
    
    
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