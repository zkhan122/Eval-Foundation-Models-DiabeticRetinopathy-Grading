import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import numpy as np

def load_auc_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["Per-class AUC"]

def class_auc_collated(json_paths, model_names, class_names, output_dir, MODE):
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                os.remove(os.path.join(output_dir, file))
                print(f"Removed: {file}")
    time.sleep(3)
    model_aucs = {}
    for path, model_name in zip(json_paths, model_names):
        model_aucs[model_name] = load_auc_data(path)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))  # class positions
    width = 0.25  # bar width
    
    color_map = {
        'CLIP': '#2ca02c',      # green
        'RETFound': '#1f77b4',  # blue
        'UrFound': '#ff7f0e'    # orange
    } 
    
    # bars for each model
    for i, model_name in enumerate(model_names):
        color = color_map[model_name]
        offset = (i - 1) * width  # -0.25, 0, 0.25
        bars = ax.bar(x + offset, model_aucs[model_name], width, 
                     label=model_name, color=color, edgecolor='black')
        # value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('AUC-ROC Score')
    ax.set_xlabel('Diabetic Retinopathy Severity')
    ax.set_title(f"{MODE} Tuned Models - AUC Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0.5, 1.0])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}")


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

    model_names = ["RETFound", "UrFound", "CLIP"]

    class_auc_collated(non_lora_jsons, model_names, dr_classes, "../plots/nonlora-final-plots", "NON-LORA")
    
    class_auc_collated(lora_jsons, model_names, dr_classes, "../plots/lora-final-plots", "LORA")
