import pandas as pd



def load_idrid_grading_labels(mode, dataset_path):
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
