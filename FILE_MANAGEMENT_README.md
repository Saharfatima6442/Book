# File Management Scripts

This directory contains four Python scripts to help manage duplicate files and create backups:

## 1. duplicate_backup.py
This script finds files with identical content in a directory and moves the duplicates to a backup folder, keeping one copy in the original location.

### Features:
- Scans a directory recursively for duplicate files based on content (not just filename)
- Creates a timestamped backup folder for duplicates
- Keeps one copy of each unique file in its original location
- Preserves the original files and only moves the redundant copies

### Usage:
```bash
python duplicate_backup.py
```

## 2. simple_duplicate_backup.py
A simplified version that focuses specifically on finding and moving duplicate files to a backup folder.

### Features:
- Finds duplicate files based on content
- Moves duplicates to a "backup_duplicates" folder
- Keeps one copy of each unique file in its original location
- Simple interface with minimal prompts

### Usage:
```bash
python simple_duplicate_backup.py
```

## 3. backup_script.py
This script creates a complete backup of all files (or filtered by extension) from a source directory to a backup location.

### Features:
- Creates a timestamped backup folder
- Copies all files from source to backup (preserving directory structure)
- Option to filter files by extension
- Preserves file metadata

### Usage:
```bash
python backup_script.py
```

## 4. duplicate_finder.py
This script identifies duplicate files but gives you the option to move them manually.

### Features:
- Finds duplicate files based on content
- Lists all duplicates before taking action
- Allows you to confirm before moving files
- Moves duplicates to a backup folder

### Usage:
```bash
python duplicate_finder.py
```

## Requirements:
- Python 3.x
- No external dependencies required

Choose the script that best fits your needs:
- Use `simple_duplicate_backup.py` for the most straightforward duplicate handling
- Use `duplicate_backup.py` if you want more control and information about the process
- Use `backup_script.py` if you want to create a full backup of your files
- Use `duplicate_finder.py` if you want more control over the duplicate removal process