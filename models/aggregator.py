import torch
import copy
from models.global_model import HeartDiseaseModel

def federated_averaging(global_model, client_weights, client_sizes):
    """
    FedAvg Algorithm - Aggregate client weights into global model
    weighted average based on number of samples each client has
    """
    total_samples = sum(client_sizes)
    
    # Initialize averaged weights with zeros
    averaged_weights = {}
    for name, param in global_model.named_parameters():
        averaged_weights[name] = torch.zeros_like(param.data)

    # Weighted average of client weights
    for client_weight, size in zip(client_weights, client_sizes):
        weight = size / total_samples
        for name in averaged_weights:
            averaged_weights[name] += client_weight[name] * weight

    # Update global model with averaged weights
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data.copy_(averaged_weights[name])

    print(f"Aggregated weights from {len(client_weights)} clients")
    print(f"Client sample sizes: {client_sizes}")
    print(f"Total samples: {total_samples}")

    return global_model

if __name__ == "__main__":
    # Test aggregation
    global_model = HeartDiseaseModel(input_size=13)
    
    # Simulate 3 client weights
    client_weights = []
    for i in range(3):
        client_model = copy.deepcopy(global_model)
        client_weights.append({
            name: param.data.clone()
            for name, param in client_model.named_parameters()
        })

    client_sizes = [266, 267, 267]
    global_model = federated_averaging(global_model, client_weights, client_sizes)
    print("\nAggregation test successful!")