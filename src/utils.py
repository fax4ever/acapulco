import torch
import random
import numpy as np
import tarfile
import os

def set_seed(seed=777, total_determinism=False):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if total_determinism:
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")

if __name__ == "__main__":
    folder_path = "/home/fax/code/acapulco/submission"            # Path to the folder you want to compress
    output_file = "/home/fax/code/acapulco/submission.gz"         # Output .gz file name
    gzip_folder(folder_path, output_file)