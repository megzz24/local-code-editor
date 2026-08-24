import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import config
import db


def create_backup(file_path: str, patch_id: int = None) -> dict:
    """Copy a file to the backup directory and record it in the DB."""
    src = Path(file_path)
    if not src.exists():
        raise IOError(f"Cannot backup non-existent file: {file_path}")

    content = src.read_bytes()
    content_hash = hashlib.md5(content).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(config.BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{src.stem}__{timestamp}__{content_hash}{src.suffix}"
    dest = backup_dir / dest_name

    try:
        shutil.copy2(src, dest)
    except OSError as e:
        raise IOError(f"Backup copy failed: {e}")

    backup_id = db.save_backup(file_path, str(dest), patch_id)
    return {"backup_id": backup_id, "backup_path": str(dest)}


def restore_backup(backup_id: int) -> dict:
    """Restore a file from a backup snapshot."""
    backups = db.get_backups()
    record = next((b for b in backups if b["id"] == backup_id), None)
    if record is None:
        return {"success": False, "restored_to": None, "error": "Backup record not found"}

    backup_path = Path(record["backup_path"])
    if not backup_path.exists():
        return {"success": False, "restored_to": None, "error": "Backup file missing from disk"}

    try:
        shutil.copy2(backup_path, record["file_path"])
    except OSError as e:
        return {"success": False, "restored_to": None, "error": str(e)}

    return {"success": True, "restored_to": record["file_path"], "error": None}


def list_backups(file_path: str = None) -> list:
    """List backups, annotating each with whether the file exists on disk."""
    records = db.get_backups(file_path)
    for r in records:
        r["exists_on_disk"] = Path(r["backup_path"]).exists()
    return records


def delete_backup(backup_id: int) -> dict:
    """Delete a backup file and its DB record."""
    backups = db.get_backups()
    record = next((b for b in backups if b["id"] == backup_id), None)
    if record is None:
        return {"success": False, "error": "Backup record not found"}

    backup_path = Path(record["backup_path"])
    if backup_path.exists():
        try:
            backup_path.unlink()
        except OSError as e:
            print(f"Warning: could not delete backup file {backup_path}: {e}")

    # Always remove DB record even if file delete failed
    with db.get_connection() as conn:
        conn.execute("DELETE FROM backups WHERE id=?", (backup_id,))

    return {"success": True, "error": None}


def prune_old_backups(keep_latest: int = config.BACKUP_RETENTION_LIMIT) -> int:
    """Delete oldest backups beyond keep_latest per file."""
    all_backups = db.get_backups()
    by_file = {}
    for b in all_backups:
        by_file.setdefault(b["file_path"], []).append(b)

    deleted = 0
    for file_path, file_backups in by_file.items():
        sorted_backups = sorted(file_backups, key=lambda b: b["created_at"], reverse=True)
        for old in sorted_backups[keep_latest:]:
            result = delete_backup(old["id"])
            if result["success"]:
                deleted += 1
    return deleted