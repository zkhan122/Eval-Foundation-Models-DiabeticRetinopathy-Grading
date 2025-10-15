import pandas as pd

BASE_PATH = "../../datasets/IDRID/B-Disease-Grading/Disease-Grading/2-Groundtruths"

def load_grading_labels(mode, dataset_path):
    labels_df = pd.read_csv(dataset_path)
    labels_df = labels_df.rename(columns=lambda x: x.strip())
    labels_df = labels_df[["Image name", "Retinopathy grade", "Risk of macular edema"]]
    labels_df = labels_df.rename(columns={"Image name": "Image_Name", "Retinopathy grade": "Retinopathy_Grade", "Risk of macular edema": "Risk_of_macular_edema"})
    if mode == "train":
        labels_df["MODE"] = "TRAIN"
    elif mode == "test":
        labels_df["MODE"] = "TEST"
    else:
        print("ERROR: INVALID MODE passed to -> load_grading_labels()")
    
    return labels_df

train_labels_df = load_grading_labels("train", f"{BASE_PATH}/IDRiD_Disease_Grading_Training_Labels.csv")
test_labels_df = load_grading_labels("test", f"{BASE_PATH}/IDRiD_Disease_Grading_Testing_Labels.csv")

print(train_labels_df)
print(test_labels_df)
