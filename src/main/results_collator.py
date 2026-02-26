import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd



def class_auc_collated(json_buffer, class_names, output_dir, MODE):

    def load_auc(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data["Per-class AUC"]

    def get_model_name(path):
        return os.path.basename(os.path.dirname(path))

    results = {}
    for path in json_buffer:
        model = get_model_name(path)
        results[model] = load_auc(path)

    num_classes = len(next(iter(results.values())))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    for class_idx in range(num_classes):

        models = []
        auc_values = []

        for model in sorted(results.keys()):
            models.append(model)
            auc_values.append(results[model][class_idx])

        plt.figure()
        plt.bar(models, auc_values)
        plt.ylim(0, 1)
        plt.ylabel("AUC")
        plt.title(f"{class_names[class_idx]} - {MODE} AUC")
        plt.tight_layout()
        save_path = os.path.join(
            output_dir,
            f"class_{class_idx}_{MODE.lower()}_auc.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":

    DR_NONLORA_TEST_RESULTS_DIR = "./testing/non-lora/results"
    DR_LORA_TEST_RESULTS_DIR = "./testing/lora-based/results"

    lora_jsons = [
        f"{DR_LORA_TEST_RESULTS_DIR}/retfound/retfound_test_results.json",
        f"{DR_LORA_TEST_RESULTS_DIR}/urfound/urfound_test_results.json",
        f"{DR_LORA_TEST_RESULTS_DIR}/clip/clip_test_results.json"]


    non_lora_jsons = [
        f"{DR_NONLORA_TEST_RESULTS_DIR}/retfound/retfound_nonlora_test_results.json",
        f"{DR_NONLORA_TEST_RESULTS_DIR}/urfound/urfound_nonlora_test_results.json",
        f"{DR_NONLORA_TEST_RESULTS_DIR}/clip/clip_nonlora_test_results.json"]


    try:
        with open(non_lora_jsons[0], 'r') as file:
            data = json.load(file)
        print("File data =", data)

    except FileNotFoundError:
        print("Error: The file was not found.")


    dr_classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
    ]

    class_auc_collated(non_lora_jsons, dr_classes, "../plots/nonlora-final-plots", "NON-LORA")
    
    class_auc_collated(lora_jsons, dr_classes, "../plots/lora-final-plots", "LORA")
