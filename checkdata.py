from PIL import Image
from glob import glob
from tqdm import tqdm
import multiprocessing

def is_image_file_valid(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image integrity
        # print(f"The PNG file is valid: {file_path}")
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid PNG file: {e} \n{file_path}")
    return False

if __name__ == "__main__":
    filepaths = glob('/data/vdj/ss/mp3d_shard_data/*/*/*/*-depth.png') # '/scratch/vdj/ss/anp_full_data-500/*/*/*.png')
    files_with_errs = []
    with multiprocessing.Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(is_image_file_valid, filepaths), total=len(filepaths)))
    files_with_errs = [path for path, result in zip(filepaths, results) if not result]
    print(files_with_errs)

    # files_with_errs = []
    # for path in tqdm(filepaths):
    #     if not is_image_file_valid(path):
    #         files_with_errs.append(path)
    # print(files_with_errs)
