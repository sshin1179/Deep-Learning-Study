import os
import subprocess

def convert_ipynb_to_pdf(notebook_file):
    # Set the path to the jupyter executable and xelatex directory
    jupyter_path = '/opt/homebrew/bin/jupyter'
    xelatex_dir = '/Library/TeX/texbin'
    
    # 1. Ensure Jupyter is Installed:
    try:
        subprocess.run([jupyter_path, '--version'], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("Error: Jupyter is not installed or not found at the specified path.")
        return

    # Check if the notebook file exists
    if not os.path.exists(notebook_file):
        print(f"Error: {notebook_file} does not exist.")
        return

    # Modify the PATH environment variable to include xelatex's directory
    os.environ['PATH'] = xelatex_dir + os.pathsep + os.environ['PATH']
    
    # 2. Convert using Full Path to Jupyter:
    result = subprocess.run([jupyter_path, 'nbconvert', '--to', 'pdf', notebook_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred during conversion: {result.stderr}")
        return

    # If everything went well
    print(f"{notebook_file} converted to PDF successfully!")
