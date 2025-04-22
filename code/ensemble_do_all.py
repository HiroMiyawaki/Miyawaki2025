# %%
import subprocess
from pathlib import Path

# %%
script_dir = Path("~/Dropbox/JupyterLab/ensemble/").expanduser()
scripts = ["ensemble_fig_1.py",
           "ensemble_fig_2.py",
           "ensemble_fig_3.py",
           "ensemble_fig_4.py",
           "ensemble_fig_s3.py",
           "ensemble_table_s1.py"
           "ensemble_table_s2.py",]

for script in scripts:
    subprocess.run(["python", str(script_dir / script)])
