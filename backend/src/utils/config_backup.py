import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigBackupRestore:
    """
    Utility class for backing up and restoring configuration settings
    """
    
    def __init__(self, backup_dir: str = "./config_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_config(self, env_file_path: str = ".env", backup_name: Optional[str] = None) -> str:
        """
        Back up the current configuration to a timestamped file
        
        Args:
            env_file_path: Path to the .env file to back up
            backup_name: Optional custom name for the backup; if not provided, uses timestamp
        
        Returns:
            Path to the created backup file
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}.json"
        
        backup_path = self.backup_dir / backup_name
        
        # Read the .env file and extract key-value pairs
        config_data = {}
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip()
        
        # Add timestamp to the backup
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "config": config_data
        }
        
        # Write the backup to file
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print(f"Configuration backed up to: {backup_path}")
        return str(backup_path)
    
    def restore_config(self, backup_file_path: str, env_file_path: str = ".env") -> bool:
        """
        Restore configuration from a backup file
        
        Args:
            backup_file_path: Path to the backup file to restore from
            env_file_path: Path where the .env file should be restored
        
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            # Read the backup file
            with open(backup_file_path, 'r') as f:
                backup_data = json.load(f)
            
            # Extract the config data
            config_data = backup_data.get("config", {})
            
            # Write the config data to the .env file
            with open(env_file_path, 'w') as f:
                for key, value in config_data.items():
                    f.write(f"{key}={value}\n")
            
            print(f"Configuration restored from: {backup_file_path}")
            return True
        except Exception as e:
            print(f"Error restoring configuration: {str(e)}")
            return False
    
    def list_backups(self) -> list:
        """
        List all available backups
        
        Returns:
            List of backup file paths
        """
        backups = []
        for file in self.backup_dir.glob("config_backup_*.json"):
            backups.append(str(file))
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return backups
    
    def delete_old_backups(self, keep_count: int = 5) -> int:
        """
        Delete old backups, keeping only the most recent ones
        
        Args:
            keep_count: Number of most recent backups to keep
        
        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()
        if len(backups) <= keep_count:
            return 0
        
        backups_to_delete = backups[keep_count:]
        deleted_count = 0
        
        for backup_path in backups_to_delete:
            try:
                os.remove(backup_path)
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting backup {backup_path}: {str(e)}")
        
        print(f"Deleted {deleted_count} old backup(s)")
        return deleted_count

# Example usage
if __name__ == "__main__":
    # Create an instance of the backup/restore utility
    config_util = ConfigBackupRestore()
    
    # Create a backup
    backup_path = config_util.backup_config()
    
    # List all backups
    print("\nAvailable backups:")
    for backup in config_util.list_backups():
        print(f"  - {backup}")
    
    # Optionally delete old backups (keeping only the 5 most recent)
    config_util.delete_old_backups(keep_count=5)