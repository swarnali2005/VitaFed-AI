import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.global_model import HeartDiseaseModel

def explain_with_shap(model, X_train, X_test, feature_names):
    """Generate SHAP explanations for the model"""

    model.eval()

    # Convert to tensor
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # Define prediction function for SHAP
    def predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            return model(x_tensor).numpy()

    print("Generating SHAP explanations...")

    # Use KernelExplainer
    explainer = shap.KernelExplainer(predict, X_train[:50])
    shap_values = explainer.shap_values(X_test[:20])

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test[:20],
        feature_names=feature_names,
        show=False
    )
    plt.title("VitaFed AI - SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to results/shap_summary.png")

    # Plot SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test[:20],
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("VitaFed AI - SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig("results/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP bar plot saved to results/shap_bar.png")

    return shap_values

if __name__ == "__main__":
    from data.heart_data import load_heart_data, split_data_for_clients

    feature_names = [
        'age', 'sex', 'chest_pain', 'resting_bp',
        'cholesterol', 'fasting_bs', 'rest_ecg',
        'max_hr', 'exercise_angina', 'st_depression',
        'st_slope', 'num_vessels', 'thal'
    ]

    df = load_heart_data()
    client_data_list = split_data_for_clients(df)

    model = HeartDiseaseModel(input_size=13)

    X_train = client_data_list[0]['X_train']
    X_test = client_data_list[0]['X_test']

    shap_values = explain_with_shap(model, X_train, X_test, feature_names)
    print("\nSHAP explanation complete!")