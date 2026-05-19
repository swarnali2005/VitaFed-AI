import torch
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from data.heart_data import load_heart_data, split_data_for_clients
from models.global_model import HeartDiseaseModel, get_model_weights
from models.aggregator import federated_averaging
from clients.client1 import train_client
from clients.client2 import train_client2
from clients.client3 import train_client3
from explainability.shap_explain import explain_with_shap
from explainability.lime_explain import explain_with_lime

FEATURE_NAMES = [
    'age', 'sex', 'chest_pain', 'resting_bp',
    'cholesterol', 'fasting_bs', 'rest_ecg',
    'max_hr', 'exercise_angina', 'st_depression',
    'st_slope', 'num_vessels', 'thal'
]

def evaluate_global_model(model, client_data_list):
    """Evaluate global model on all clients test data"""
    model.eval()
    all_accuracies = []

    with torch.no_grad():
        for client_data in client_data_list:
            X_test = torch.FloatTensor(client_data['X_test'])
            y_test = torch.FloatTensor(client_data['y_test'])
            outputs = model(X_test).squeeze()
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            all_accuracies.append(accuracy.item())

    global_accuracy = np.mean(all_accuracies)
    return global_accuracy, all_accuracies

def plot_training_results(round_accuracies, client_accuracies_per_round):
    """Plot federated learning training results"""
    plt.figure(figsize=(12, 5))

    # Global accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(round_accuracies) + 1),
             round_accuracies, 'b-o', linewidth=2)
    plt.title('VitaFed AI - Global Model Accuracy')
    plt.xlabel('Federated Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)

    # Client accuracies plot
    plt.subplot(1, 2, 2)
    client_colors = ['red', 'green', 'blue']
    for i in range(3):
        client_acc = [round_acc[i]
                      for round_acc in client_accuracies_per_round]
        plt.plot(range(1, len(client_acc) + 1),
                 client_acc, color=client_colors[i],
                 marker='o', label=f'Client {i+1}')
    plt.title('VitaFed AI - Client Accuracies per Round')
    plt.xlabel('Federated Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training results plot saved to results/training_results.png")

def main():
    print("=" * 60)
    print("   VitaFed AI - Federated Learning for Heart Disease")
    print("=" * 60)

    # Load and split data
    print("\n[1] Loading and splitting data...")
    df = load_heart_data()
    client_data_list = split_data_for_clients(df)

    # Initialize global model
    print("\n[2] Initializing global model...")
    global_model = HeartDiseaseModel(input_size=13)
    print("Global model initialized!")

    # Federated learning rounds
    num_rounds = 5
    round_accuracies = []
    client_accuracies_per_round = []
    results = {}

    print(f"\n[3] Starting Federated Learning for {num_rounds} rounds...")
    print("-" * 60)

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Federated Round {round_num}/{num_rounds} ---")

        # Train each client
        client_results = []
        train_functions = [train_client, train_client2, train_client3]

        for i, (train_fn, client_data) in enumerate(
                zip(train_functions, client_data_list)):
            print(f"\nTraining Client {i+1}...")
            result = train_fn(client_data, global_model, epochs=10)
            client_results.append(result)

        # Aggregate weights
        print(f"\nAggregating weights...")
        client_weights = [r['weights'] for r in client_results]
        client_sizes = [r['size'] for r in client_results]
        global_model = federated_averaging(
            global_model, client_weights, client_sizes)

        # Evaluate global model
        global_acc, client_accs = evaluate_global_model(
            global_model, client_data_list)
        round_accuracies.append(global_acc)
        client_accuracies_per_round.append(client_accs)

        print(f"\nRound {round_num} Global Accuracy: {global_acc:.4f}")
        for i, acc in enumerate(client_accs):
            print(f"  Client {i+1} Accuracy: {acc:.4f}")

        results[f'round_{round_num}'] = {
            'global_accuracy': global_acc,
            'client_accuracies': client_accs
        }

    # Save results
    print("\n[4] Saving results...")
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to results/training_results.json")

    # Plot results
    plot_training_results(round_accuracies, client_accuracies_per_round)

    # Save global model
    torch.save(global_model.state_dict(), 'results/global_model.pth')
    print("Global model saved to results/global_model.pth")

    # Generate explanations
    print("\n[5] Generating XAI Explanations...")
    X_train = client_data_list[0]['X_train']
    X_test = client_data_list[0]['X_test']

    explain_with_shap(global_model, X_train, X_test, FEATURE_NAMES)
    explain_with_lime(global_model, X_train, X_test, FEATURE_NAMES)

    print("\n" + "=" * 60)
    print("   VitaFed AI Training Complete!")
    print(f"   Final Global Accuracy: {round_accuracies[-1]:.4f}")
    print("   Check results/ folder for all outputs!")
    print("=" * 60)

if __name__ == "__main__":
    main()