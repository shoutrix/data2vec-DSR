import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class StudentTeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.config = config
        # self.model_name = config.model_name
        model_name="facebook/wav2vec2-large-lv60"
        self.teacher = Wav2Vec2Model.from_pretrained(model_name)
        
        # print(self.teacher)
        
        self.student = Wav2Vec2Model.from_pretrained(model_name)

        self.teacher.feature_extractor.requires_grad_(False)
        self.student.feature_extractor.requires_grad_(False)
        self.teacher.encoder.requires_grad_(False)

        self.criterion = nn.MSELoss()

    def forward(self, student_input, teacher_input):
        # student_input : shape : (B, N_samples)
        # teacher_input : shape : (B, N_samples)
        with torch.no_grad():
            teacher_features = self.teacher.feature_extractor(teacher_input)
            student_features = self.student.feature_extractor(student_input)
            teacher_features = self.match_length(teacher_features, student_features)
            teacher_output = self.teacher.encoder(teacher_features, output_hidden_states=True).hidden_states
        student_outputs = self.student.encoder(student_features, output_hidden_states=True).hidden_states
  
    def match_length(self, teacher_features, student_features):
        return teacher_features
  
  
  
  
        
# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentTeacherModel().to(device)
optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-5)

# Dummy tensors to simulate audio input
student_audio = torch.randn(2, 48000).to(device)  # Batch of 2, 1 sec of audio (16kHz)
teacher_audio = torch.randn(2, 16000).to(device)  # Corresponding regular speech

# Training step
model.train()
optimizer.zero_grad()
loss = model(student_audio, teacher_audio)
# loss.backward()
# optimizer.step()

print(f"Training Loss: {loss.item()}")
