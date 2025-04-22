import joblib
import re
from pathlib import Path
    
def get_rootdir():
    return Path('~/Dropbox/analysesPython/data/').expanduser()

def get_rats(data_dir=str(get_rootdir())):
    data_path=Path(data_dir)
    return sorted([p.name for p in data_path.glob('*') if re.search('.+\d{6}$', str(p))])
