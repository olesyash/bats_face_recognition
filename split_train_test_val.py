import shutil

from sklearn.model_selection import train_test_split
import os

# input_folder = r"C:\olesya\bats_frames\3"

output_folder = r"C:\olesya\bats_new"


def load_data_from_folder(folder_path):
    images = os.listdir(folder_path)
    return images

def run_for_folder(input_folder):
    f_name = os.path.basename(input_folder)
    output_path = os.path.join(output_folder, f_name)
    print(output_path)
    data = load_data_from_folder(input_folder)
    # Assuming `data` and `labels` are your dataset and labels
    train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=True, random_state=42)
    print(train_data)
    train_folder = os.path.join(output_folder, 'train', f_name)
    os.makedirs(train_folder, exist_ok=True)
    # Split the temporary set into validation and testing sets
    val_data, test_data, = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=42)
    print(val_data)
    test_folder = os.path.join(output_folder, 'test', f_name)
    os.makedirs(test_folder, exist_ok=True)
    print(test_data)
    val_folder = os.path.join(output_folder, 'validation', f_name)
    os.makedirs(val_folder, exist_ok=True)

    # copy the files to the new folders
    for image in train_data:
        src = os.path.join(input_folder, image)
        dst = os.path.join(train_folder, image)
        # print("Copying", src, "to", dst)
        shutil.copy(src, dst)

        # copy the files to the new folders
    for image in val_data:
        src = os.path.join(input_folder, image)
        dst = os.path.join(val_folder, image)
        # print("Copying", src, "to", dst)
        shutil.copy(src, dst)

    for image in test_data:
        src = os.path.join(input_folder, image)
        dst = os.path.join(test_folder, image)
        # print("Copying", src, "to", dst)
        shutil.copy(src, dst)


all_folder_path = r"C:\olesya\bats_frames"
all_folders = os.listdir(all_folder_path)
for input_folder in all_folders:
    if input_folder not in ["test", "train"]:
        input_folder = os.path.join(all_folder_path, input_folder)
        run_for_folder(input_folder)

