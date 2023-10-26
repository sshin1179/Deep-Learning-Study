def convert_ipynb_to_pdf(notebook_file):
    import os
    import subprocess

    # Set the path to the jupyter executable and xelatex directory
    # jupyter_path = '/opt/homebrew/bin/jupyter'
    # xelatex_dir = '/Library/TeX/texbin'
    try:
        os.environ['PATH'] = '/Library/TeX/texbin' + os.pathsep + os.environ['PATH']
        subprocess.run(['/opt/homebrew/bin/jupyter', 'nbconvert', '--to', 'pdf', notebook_file], capture_output=True, text=True)
    except:
        print('Fix Environment Variables.')
        pass