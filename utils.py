# utils.py - Helper functions for Agentic AI Project
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Setup logging
def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# Token counting estimate
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# Save results to JSON
def save_to_json(data: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved to {filepath}")

# Load JSON file
def load_from_json(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return json.load(f)

# Format agent response
def format_response(agent_output: str, metadata: Dict = None) -> Dict:
    return {
        "response": agent_output,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }

# Chunk text for vector storage
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

logger = setup_logger(__name__)
