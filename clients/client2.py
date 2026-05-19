import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models.global_model import HeartDiseaseModel, get_model_weights

def train_client2(client_data, global_model, epochs=10, learning_rate=0.01):
    """Train model on client 2's local data"""

    # Deep copy global model for local training
    local_model = copy.deepcopy(global_model)
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Convert data to tensors
    X_train = torch.FloatTensor(client_data['X_train'])
    y_train = torch.FloatTensor(client_data['y_train']).unsqueeze(1)

    local_model.train()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = local_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"  Client {client_data['client_id']} - "
                  f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {loss.item():.4f}")

    # Evaluate on test data
    local_model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(client_data['X_test'])
        y_test = torch.FloatTensor(client_data['y_test'])
        outputs = local_model(X_test).squeeze()
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f"  Client {client_data['client_id']} - "
              f"Test Accuracy: {accuracy.item():.4f}")

    return {
        'weights': get_model_weights(local_model),
        'size': len(client_data['X_train']),
        'accuracy': accuracy.item(),
        'losses': losses
    }

if __name__ == "__main__":
    from data.heart_data import load_heart_data, split_data_for_clients

    df = load_heart_data()
    client_data_list = split_data_for_clients(df)
    global_model = HeartDiseaseModel(input_size=13)

    print("\nTraining Client 2...")
    result = train_client2(client_data_list[1], global_model)
    print(f"\nClient 2 Training Complete!")
    print(f"Final Accuracy: {result['accuracy']:.4f}")