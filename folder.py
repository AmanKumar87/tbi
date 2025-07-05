# create_nested_label_folders.py
import os

def create_nested_folders(base_dir="data"):
    labels = [
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "any"
    ]

    try:
        for label in labels:
            for sub in ['0', '1']:  # 0 = not having, 1 = having
                path = os.path.join(base_dir, label, sub)
                os.makedirs(path, exist_ok=True)
        print("✅ Folder structure created successfully with nested '0' and '1' folders.")
    except Exception as e:
        print(f"\033[93m⚠️ Error creating folders: {e}\033[0m")

if __name__ == "__main__":
    create_nested_folders()
