import os
import shutil
from datetime import datetime

def create_backup_folder(base_path="."):
    """Create a backup folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(base_path, f"backup_{timestamp}")
    os.makedirs(backup_folder, exist_ok=True)
    return backup_folder

def backup_files(source_dir, backup_dir, file_extensions=None):
    """Backup files from source directory to backup directory."""
    files_backed_up = 0
    
    for root, dirs, files in os.walk(source_dir):
        # Skip hidden directories like .git, .specify, etc.
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip the script file itself and directories
            if os.path.isdir(file_path) or file == 'backup_script.py':
                continue
                
            # If file extensions filter is provided, only backup matching files
            if file_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() not in file_extensions:
                    continue
            
            # Determine the relative path from source directory
            rel_path = os.path.relpath(file_path, source_dir)
            backup_path = os.path.join(backup_dir, rel_path)
            
            # Create directory structure in backup if needed
            backup_file_dir = os.path.dirname(backup_path)
            os.makedirs(backup_file_dir, exist_ok=True)
            
            # Copy the file to backup location
            shutil.copy2(file_path, backup_path)
            print(f"Backed up: {file_path} -> {backup_path}")
            files_backed_up += 1
    
    return files_backed_up

def main():
    print("File Backup Script")
    print("==================")
    
    source_dir = input("Enter the source directory path (press Enter for current directory): ").strip()
    if not source_dir:
        source_dir = "."
    
    if not os.path.isdir(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Ask for file extensions filter
    ext_input = input("Enter file extensions to backup (comma-separated, e.g., '.txt,.py,.md'; press Enter for all files): ").strip()
    file_extensions = None
    if ext_input:
        file_extensions = [ext.strip().lower() for ext in ext_input.split(',')]
    
    # Create backup folder
    backup_dir = create_backup_folder(source_dir)
    print(f"Creating backup in: {backup_dir}")
    
    # Perform backup
    files_count = backup_files(source_dir, backup_dir, file_extensions)
    
    print(f"\nBackup completed!")
    print(f"Files backed up: {files_count}")
    print(f"Backup location: {backup_dir}")

if __name__ == "__main__":
    main()