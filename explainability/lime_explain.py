import lime
import lime.lime_tabular
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.global_model import HeartDiseaseModel

def explain_with_lime(model, X_train, X_test, feature_names):
    """Generate LIME explanations for the model"""

    model.eval()

    # Define prediction function for LIME
    def predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            probs = model(x_tensor).numpy()
            return np.hstack([1 - probs, probs])

    print("Generating LIME explanations...")

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        mode='classification'
    )

    # Explain first 3 test samples
    for i in range(3):
        exp = explainer.explain_instance(
            X_test[i],
            predict,
            num_features=13
        )

        # Save explanation plot
        fig = exp.as_pyplot_figure()
        plt.title(f"VitaFed AI - LIME Explanation (Sample {i+1})")
        plt.tight_layout()
        plt.savefig(f"results/lime_explanation_{i+1}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"LIME explanation {i+1} saved to results/lime_explanation_{i+1}.png")

    return explainer

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

    explain_with_lime(model, X_train, X_test, feature_names)
    print("\nLIME explanation complete!")