import os
import json

def save_to_jsonl(content: str, filename:str) -> str:
    """
    Saves a string to a JSONL file.
    Logic: If file exists -> delete it -> Create a new file -> Write to it
    """
    # 1.To Ensure the folder exists
    folder = "training_data"
    os.makedirs(folder, exist_ok=True)

    # 2. Construct the file path
    clean_name = filename.strip().replace(" ", "_")
    if not clean_name.endswith(".jsonl"):
        clean_name += ".jsonl"

    file_path = os.path.join(folder, clean_name)


    # 3. Delete the file if it exists
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"🗑️ FOUND OLD FILE. DELETED: {file_path}")
        except OSError as e:
            print(f"⚠️ Warning: Could not delete old file: {e}")

    
    # 4. Write to the new file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ NEW FILE CREATED: {file_path}")
        return file_path

    except Exception as e:
        print(f"⚠️ Warning: Could not write to new file: {e}")
        return "Error: Could not write to new file"
    
