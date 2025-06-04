# IR_classification_WBC
## How to Use

### **Run the Scripts**

There are two main steps in the workflow:

#### Step 1: Data Preprocessing

- Open and run `function/preprocessing_script.ipynb`.
- This script loads the raw data from the `data/rawdata` folder (doi: 10.5281/zenodo.15589352) and applies several preprocessing steps.

The preprocessing script makes use of the following helper modules:

- `preprocess.py`, `PreprocSpec.py`, `UtilsSpec.py`:  
  Implement normalization, smoothing, baseline correction, and other preprocessing techniques.

#### Step 2: Classification Modeling

- Run `model_script.ipynb` to train and evaluate classification models using the preprocessed data from the previous step.
- The notebook demonstrates how to call the main modeling function implemented in `fullmodel.py`.

Additional supporting modules:

- `combinations.py`:  
  Defines combinations of different cell types for classification tasks.

- `ModelSpec.py`, `modeltransfer.py`:  
  Required for using the Extended Multiplicative Signal Correction (EMSC) technique.

- `pca_visualization.py`, `pltspectra.py`:  
  Visualization utilities for PCA plots and spectral plots.


---

## Requirements

- Python 3.x
- Recommended environment: Jupyter Notebook

### Required Python packages:
- `numpy`  
- `pandas`  
- `matplotlib`  
- (Other dependencies are specified and imported directly in the main notebooks.)
