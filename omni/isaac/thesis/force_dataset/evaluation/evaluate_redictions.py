import json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def get_labels():
    # Example true and predicted labels
    true_labels = [1, 0, 1, 1, 0, 1, 0]
    predicted_labels = [1, 0, 1, 1, 1, 0, 1]

    # Create a dictionary to store the labels
    labels_dict = {
        "true_labels": true_labels,
        "predicted_labels_LSTM": predicted_labels
        # "predicted_labels_SVM": predicted_labels
        # "predicted_labels_RF": predicted_labels
        # "predicted_labels_DTW": predicted_labels
    }

    # Write the labels to a JSON file
    with open("labels.json", "w") as json_file:
        json.dump(labels_dict, json_file, indent=4)

    print("Labels written to labels.json")


def read_labels():
    # Read the labels from the JSON file
    with open("labels.json", "r") as json_file:
        labels_dict = json.load(json_file)

    # Retrieve the true labels and predicted labels
    true_labels = labels_dict["true_labels"]
    predicted_labels = labels_dict["predicted_labels"]

    # Print the labels
    print("True Labels:", true_labels)
    print("Predicted Labels:", predicted_labels)
    
    
def evaluation(true_labels, predicted_labels):
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Create a dictionary to store the metrics
    metrics_dict = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }