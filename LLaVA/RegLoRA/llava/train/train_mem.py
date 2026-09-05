import sys
from pathlib import Path
_METHOD_ROOT = str(Path(__file__).resolve().parents[2])
if _METHOD_ROOT not in sys.path:
    sys.path.append(_METHOD_ROOT)
from llava.train.train import train

if __name__ == "__main__":
    train()
