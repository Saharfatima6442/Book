import os
import hashlib
from collections import defaultdict
import shutil

def calculate_file_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_and_backup_duplicates(directory, backup_dir="backup_duplicates"):
    """Find duplicate files and move them to a backup directory."""
    hash_to_files = defaultdict(list)
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip directories and this script file
            if os.path.isdir(file_path) or file == 'simple_duplicate_backup.py':
                continue
                
            file_hash = calculate_file_hash(file_path)
            hash_to_files[file_hash].append(file_path)
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    duplicates_found = 0
    
    # Process each group of duplicate files
    for file_hash, file_list in hash_to_files.items():
        if len(file_list) > 1:  # More than one file with the same content
            # Keep the first file, move the rest to backup
            original_file = file_list[0]
            duplicates = file_list[1:]
            
            for dup in duplicates:
                # Create a unique name in the backup folder
                file_name = os.path.basename(dup)
                backup_path = os.path.join(backup_dir, file_name)
                
                # Handle naming conflicts in the backup folder
                counter = 1
                original_backup_path = backup_path
                while os.path.exists(backup_path):
                    name, ext = os.path.splitext(original_backup_path)
                    backup_path = f"{name}_duplicate_{counter}{ext}"
                    counter += 1
                
                # Move the duplicate file to the backup folder
                shutil.move(dup, backup_path)
                print(f"Moved duplicate: {dup} -> {backup_path}")
                duplicates_found += 1
    
    return duplicates_found

def main():
    print("Simple Duplicate Files Backup")
    print("=============================")
    
    directory = input("Enter directory to scan for duplicates (Enter for current): ").strip() or "."
    
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    print(f"\nScanning for duplicates in: {directory}")
    num_duplicates = find_and_backup_duplicates(directory)
    
    if num_duplicates > 0:
        print(f"\nSuccessfully moved {num_duplicates} duplicate files to backup_duplicates folder.")
        print("Original files remain in their locations.")
    else:
        print("\nNo duplicates found.")

if __name__ == "__main__":
    main()