import os

def root_dir():
    """Returns the full path of the root directory of the project."""
    
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return root_dir
    
def full_data_path(filename):
    """Returns the full path of filename (str) located in the data folder of the project.
    """
    
    return os.path.join(root_dir(), 'data', filename)

def full_models_path(filename):
    """Returns the full path of filename (str) located in the data folder of the project.
    """
    return os.path.join(root_dir(), 'models', filename)