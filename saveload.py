import pickle

def save(obj, file):
    """
    :param obj: object to be saved
    :param file: file name
    """
    with open(file, 'wb') as f:
       pickle.dump(obj, f, protocol=4)

def load(file):
    """
    :param file: file name
    :return: the object read
    """
    with open(file, 'rb') as f:
        return pickle.load(f)
