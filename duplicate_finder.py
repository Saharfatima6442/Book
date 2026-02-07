import os
import hashlib
from pathlib import Path

def calculate_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def find_duplicates(directory):
    """Find duplicate files in a directory based on content."""
    hashes = {}
    duplicates = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories like .git, .specify, etc.
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip the script file itself and directories
            if os.path.isdir(file_path) or file == 'duplicate_finder.py':
                continue
                
            file_hash = calculate_file_hash(file_path)
            
            if file_hash in hashes:
                # This is a duplicate
                duplicates.append({
                    'original': hashes[file_hash],
                    'duplicate': file_path,
                    'hash': file_hash
                })
            else:
                # This is the first occurrence of this content
                hashes[file_hash] = file_path
    
    return duplicates

def move_duplicate_to_backup(duplicate_path, backup_dir):
    """Move a duplicate file to the backup directory."""
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create a unique name for the duplicate in the backup
    file_name = os.path.basename(duplicate_path)
    backup_path = os.path.join(backup_dir, file_name)
    
    # If a file with the same name already exists in backup, add a counter
    counter = 1
    original_backup_path = backup_path
    while os.path.exists(backup_path):
        name, ext = os.path.splitext(original_backup_path)
        backup_path = f"{name}_{counter}{ext}"
        counter += 1
    
    # Move the file to backup
    os.rename(duplicate_path, backup_path)
    print(f"Moved duplicate: {duplicate_path} -> {backup_path}")
    return backup_path

def main():
    directory = input("Enter the directory path to scan for duplicates (press Enter for current directory): ").strip()
    if not directory:
        directory = "."
    
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    print(f"Scanning for duplicates in: {directory}")
    duplicates = find_duplicates(directory)
    
    if not duplicates:
        print("No duplicates found.")
        return
    
    print(f"\nFound {len(duplicates)} duplicate files:")
    for dup in duplicates:
        print(f"  Original: {dup['original']}")
        print(f"  Duplicate: {dup['duplicate']}")
        print(f"  Hash: {dup['hash'][:10]}...")
        print()
    
    backup_dir = os.path.join(directory, "backup_duplicates")
    
    response = input(f"Do you want to move the duplicates to '{backup_dir}'? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        for dup in duplicates:
            move_duplicate_to_backup(dup['duplicate'], backup_dir)
        print(f"\nDuplicates moved to: {backup_dir}")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()