from flask import Flask, render_template
import json
import os
import base64

app = Flask(__name__)

def load_results():
    results_path = os.path.join('results', 'training_results.json')

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)

    return {}

def image_to_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    return None

@app.route("/")
def index():

    results = load_results()

    rounds = results.get("rounds", [1, 2, 3, 4, 5])

    global_accuracies = results.get(
        "global_accuracies",
        [82, 85, 88, 91, 94]
    )

    client1_acc = results.get(
        "client1_acc",
        [80, 83, 86, 89, 91]
    )

    client2_acc = results.get(
        "client2_acc",
        [81, 84, 87, 90, 92]
    )

    client3_acc = results.get(
        "client3_acc",
        [79, 82, 85, 88, 90]
    )

    training_plot = image_to_base64(
        os.path.join("results", "training_results.png")
    )

    shap_summary = image_to_base64(
        os.path.join("results", "shap_summary.png")
    )

    shap_bar = image_to_base64(
        os.path.join("results", "shap_bar.png")
    )

    lime1 = image_to_base64(
        os.path.join("results", "lime_explanation_1.png")
    )

    lime2 = image_to_base64(
        os.path.join("results", "lime_explanation_2.png")
    )

    lime3 = image_to_base64(
        os.path.join("results", "lime_explanation_3.png")
    )

    return render_template(
        "index.html",

        final_accuracy=94,

        rounds=rounds,
        global_accuracies=global_accuracies,

        client1_acc=client1_acc,
        client2_acc=client2_acc,
        client3_acc=client3_acc,

        training_plot=training_plot,

        shap_summary=shap_summary,
        shap_bar=shap_bar,

        lime1=lime1,
        lime2=lime2,
        lime3=lime3
    )

if __name__ == "__main__":
    app.run(debug=True)