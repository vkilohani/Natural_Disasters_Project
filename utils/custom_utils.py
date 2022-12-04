import os

def root_dir():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return root_dir
    
def full_data_path(filename):
    """Gets full path of the files in the given folder
    args:
    - filename
    returns:
    full path of the filename on the system
    """

    return os.path.join(root_dir(), 'data', filename)

def full_models_path(filename):
    """Gets full path of the files in the models folder
    args:
    - filename
    returns:
    full path of the filename on the system
    """
    return os.path.join(root_dir(), 'models', filename)