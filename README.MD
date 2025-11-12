# NeuroMarketing Repository

## Getting Started

To get started with this repository, follow the steps below:

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    ```

2. **Install Dependencies**
    Make sure to download the `ipynb-py-convert` library:
    ```bash
    pip install ipynb-py-convert
    ```

3. **Convert Python Script to Jupyter Notebook**
    Run the following command to convert the Python script to a Jupyter Notebook:
    ```bash
    ipynb-py-convert PreProcessing-SEED-IV.py PreProcessing-SEED-IV.ipynb
    ```

4. **Make Your Changes**
    Edit the Jupyter Notebook as needed.

5. **Convert Back to Python Script**
    After making changes, convert the Jupyter Notebook back to a Python script:
    ```bash
    ipynb-py-convert PreProcessing-SEED-IV.ipynb PreProcessing-SEED-IV.py
    ```

6. **Commit and Push Changes**
    Finally, commit your changes and push them to the repository:

## Important Note
Please avoid pushing `.ipynb` files to the repository, as merging and reading the file differences can be quite challenging.