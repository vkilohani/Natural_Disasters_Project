import os

def full_data_path(filename):
    """Gets full path of the files in the given folder
    args:
    - filename
    returns:
    full path of the filename on the system
    """
    return os.path.join(os.getcwd(), 'data', filename)

def full_models_path(filename):
    """Gets full path of the files in the models folder
    args:
    - filename
    returns:
    full path of the filename on the system
    """
    return os.path.join(os.getcwd(), 'models', filename)