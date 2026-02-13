# services/ocr_cache.py
import os, json, hashlib
from typing import Optional, Dict, Any

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class OCRCache:
    def __init__(self, cache_dir: str = "ocr_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def cache_path(self, image_path: str) -> str:
        key = sha256_file(image_path)
        return os.path.join(self.cache_dir, f"{key}.json")

    def load(self, image_path: str) -> Optional[Dict[str, Any]]:
        p = self.cache_path(image_path)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, image_path: str, ocr_result: Dict[str, Any]) -> str:
        p = self.cache_path(image_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        return p
