import os
import shutil
import random

def extract_for_split(source_folder, destination_folder, test_fraction):

    all_files = os.listdir(source_folder)
    image_files = [f for f in all_files if os.path.isfile(os.path.join(source_folder, f))]
    
    total_images = len(image_files)
    
    if total_images == 0:
        print("Error: No images found in the source folder '{}'.".format(source_folder))
        return

    test_size = int(total_images * test_fraction) 

    if os.path.exists(destination_folder):
        all_files_dest = os.listdir(destination_folder)
        image_files_dest = [f for f in all_files_dest if os.path.isfile(os.path.join(destination_folder, f))]
        current_test_size = len(image_files_dest)
        
        # We use a small tolerance (e.g., 5 images) for robustness
        tolerance = 5 
        
        if abs(current_test_size - test_size) <= tolerance:
            print("---------------------------------------------------------------------------------------")
            print("Skipping split: Test folder already exists and contains a near-expected number of files.")
            print("Found {} files (Expected: ~{}).".format(current_test_size, test_size))
            print("---------------------------------------------------------------------------------------")
            return
    
    print("Found {} total images.".format(total_images))
    print("Calculating test size as {}% ({} images).".format(
        int(test_fraction * 100), test_size))
    
    os.makedirs(destination_folder, exist_ok=True)
    print("Destination folder '{}' ensured.".format(destination_folder))
    
    selected_files = random.sample(image_files, test_size)
    
    for filename in selected_files:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(source_path, destination_path) # to make sure the images going to the test dataset do not remain original

    print("Successfully copied {} images to '{}'.".format(len(selected_files), destination_folder))



extract_for_split("/vol/research/ZayaanProject2025-26/Eval-Foundation-Models-DiabeticRetinopathy-Grading/datasets/EYEPACS/train", "/vol/research/ZayaanProject2025-26/Eval-Foundation-Models-DiabeticRetinopathy-Grading/datasets/EYEPACS/test", 0.1)
