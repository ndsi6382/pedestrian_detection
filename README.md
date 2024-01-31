# Local Python Files

* Requirements for these files can be installed with 
`pip install -r requirements.txt`.

* All materials for this project (except for the dataset) must be located in the
same directory. It will be assumed that these files will be located in the same
folder as the trained models, and all pictorial outputs will be written to the
same directory.

* Each `.py` file must be run from the directory in which it is placed (e.g. 
`python eval.py`, not `python3 abc/def/eval.py`), i.e. `cd` into the directory 
first.

* The dataset folder structure must not be changed.

* The paths to the dataset on the local computer must be set in 
`project_requirements.py` with the `IMAGE_PATH` and `LABEL_PATH` variables.
