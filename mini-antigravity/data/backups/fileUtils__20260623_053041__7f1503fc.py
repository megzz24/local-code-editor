import os
import json

def read_file(path):
    """Read a file and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    """Write a string to a file, creating it if it doesn't exist."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def load_json(path):
    """Load and return a JSON file as a Python dict."""
    with open(path, "r") as f:
        return json.load(f)

def list_files(directory, extension=None):
    """List all files in a directory, optionally filtered by extension."""
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return sorted(files)

class FileCache:
    """Simple in-memory cache for file contents."""

    def __init__(self):
        self._cache = {}

    def get(self, path):
        """Return cached file content or None if not cached."""
        return self._cache.get(path)

    def set(self, path, content):
        """Store file content in the cache."""
        self._cache[path] = content

    def clear(self):
        """Clear all cached files."""
        self._cache.clear()