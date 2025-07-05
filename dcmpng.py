import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_dcm_to_png(dcm_path, png_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img_array = ds.pixel_array

        # Normalize to 0-255
        img_array = img_array.astype(float)
        img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_array = np.uint8(img_array)

        img = Image.fromarray(img_array)
        img.save(png_path)
        return True
    except Exception as e:
        print(f"\033[93m⚠️ Failed to convert {dcm_path}: {e}\033[0m")
        return False

def convert_all_dcm_to_png():
    print("🔄 Searching and converting .dcm files in current directory and subfolders...")

    converted_count = 0

    while True:
        dcm_files = []

        # Recursively find all .dcm files from current directory
        for root, _, files in os.walk(os.getcwd()):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dcm_files.append(os.path.join(root, file))

        if not dcm_files:
            print("\033[92m✅ All DICOM files converted.\033[0m")
            break

        for dcm_file in tqdm(dcm_files, desc="Converting"):
            png_file = dcm_file.replace(".dcm", ".png")
            if convert_dcm_to_png(dcm_file, png_file):
                os.remove(dcm_file)
                print(f"{os.path.basename(dcm_file)} \033[92m✅ Converted & removed\033[0m")
                converted_count += 1

    print(f"\n🎉 Total files converted: {converted_count}")

if __name__ == "__main__":
    convert_all_dcm_to_png()
