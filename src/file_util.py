import os, pickle
from src.loadData import GraphDataset

def dump(file_name, result):
    # remove dump files if present
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as file:
        print("dumping", file_name)
        # noinspection PyTypeChecker
        pickle.dump(result, file)

def load(file_name):
    with open(file_name, 'rb') as file:
        print("loading", file_name)
        # noinspection PyTypeChecker
        return pickle.load(file)

def get_or_create_graph_ds(binary_file_name, json_file_name, transform):
    base_dir = os.path.dirname(os.path.abspath(json_file_name))
    binary_file_name = os.path.join(base_dir, binary_file_name)

    if not (os.path.exists(binary_file_name)):
        dataset = GraphDataset(json_file_name, transform=transform)
        dump(binary_file_name, dataset)
    else:
        dataset = load(binary_file_name)
    return dataset