import os
import hashlib
from collections import defaultdict
import shutil
from datetime import datetime

def calculate_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def find_duplicate_groups(directory):
    """Find groups of duplicate files in a directory based on content."""
    hash_to_files = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories like .git, .specify, etc.
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip the script file itself and directories
            if os.path.isdir(file_path) or file in ['duplicate_backup.py', 'backup_script.py', 'duplicate_finder.py']:
                continue
                
            file_hash = calculate_file_hash(file_path)
            hash_to_files[file_hash].append(file_path)
    
    # Return only groups with more than one file (duplicates)
    duplicate_groups = [group for group in hash_to_files.values() if len(group) > 1]
    return duplicate_groups

def create_backup_folder(base_path="."):
    """Create a backup folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(base_path, f"backup_{timestamp}")
    os.makedirs(backup_folder, exist_ok=True)
    return backup_folder

def move_duplicates_to_backup(duplicate_groups, backup_dir):
    """Move duplicate files to the backup directory, keeping one copy in original location."""
    total_moved = 0
    
    for group in duplicate_groups:
        # Keep the first file in the original location, move the rest to backup
        original_file = group[0]
        duplicates_to_move = group[1:]
        
        for dup in duplicates_to_move:
            # Create a unique name for the duplicate in the backup
            file_name = os.path.basename(dup)
            backup_path = os.path.join(backup_dir, file_name)
            
            # If a file with the same name already exists in backup, add a counter
            counter = 1
            original_backup_path = backup_path
            while os.path.exists(backup_path):
                name, ext = os.path.splitext(original_backup_path)
                backup_path = f"{name}_copy_{counter}{ext}"
                counter += 1
            
            # Move the file to backup
            shutil.move(dup, backup_path)
            print(f"Moved duplicate: {dup} -> {backup_path}")
            total_moved += 1
    
    return total_moved

def main():
    print("Duplicate Files Backup Script")
    print("=============================")
    
    directory = input("Enter the directory path to scan for duplicates (press Enter for current directory): ").strip()
    if not directory:
        directory = "."
    
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    print(f"\nScanning for duplicates in: {directory}")
    duplicate_groups = find_duplicate_groups(directory)
    
    if not duplicate_groups:
        print("No duplicates found.")
        return
    
    print(f"\nFound {len(duplicate_groups)} groups of duplicate files:")
    total_duplicates = sum(len(group) - 1 for group in duplicate_groups)  # -1 to exclude the original from count
    print(f"Total duplicate files to backup: {total_duplicates}")
    
    for i, group in enumerate(duplicate_groups):
        print(f"  Group {i+1}: {len(group)} identical files")
        for j, file_path in enumerate(group):
            marker = "(kept)" if j == 0 else "(moved to backup)"
            print(f"    {marker} {file_path}")
        print()
    
    backup_dir = create_backup_folder(directory)
    
    response = input(f"\nDo you want to move duplicates (keeping one copy of each) to '{backup_dir}'? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        total_moved = move_duplicates_to_backup(duplicate_groups, backup_dir)
        print(f"\nDuplicates moved to: {backup_dir}")
        print(f"Total files moved: {total_moved}")
        print("\nThe original files remain in their locations, and duplicates are now in the backup folder.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()