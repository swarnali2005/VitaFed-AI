import torch
import torch.nn as nn

class HeartDiseaseModel(nn.Module):
    """Neural Network model for Heart Disease Prediction"""

    def __init__(self, input_size=13):
        super(HeartDiseaseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def get_model_weights(model):
    """Extract model weights"""
    return {name: param.data.clone()
            for name, param in model.named_parameters()}

def set_model_weights(model, weights):
    """Set model weights"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])

if __name__ == "__main__":
    model = HeartDiseaseModel(input_size=13)
    print("VitaFed AI - Heart Disease Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")