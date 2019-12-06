from pathlib import Path

DATA_DIR = Path.cwd() / 'data'
MODEL_DIR = DATA_DIR / 'models'

for directory in [DATA_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)
