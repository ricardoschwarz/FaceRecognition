import argparse
import os
import glob
import time
import multiprocessing as mp

import preprocessing

def main(input_dir, output_dir):
    """
    Main logic to load, process and save all images of the dataset. 
    """
    image_path_gen = (path for path in glob.iglob(input_dir + "**/*.jpg"))
    pool = mp.Pool(processes=mp.cpu_count())
    print(f"Using {mp.cpu_count()} CPU Cores")

    start = time.time()
    for image_path in image_path_gen:
        output_path = image_path.replace(input_dir, output_dir)
        pool.apply_async(preprocessing.preprocess, (image_path, output_path))

    pool.close()
    pool.join()
    end = time.time()

    print(end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess lfw data set.")
    parser.add_argument("--input_dir", metavar="", type=str, action="store", default=".\\lfw\\", dest="input_dir", help="directory of the lfw dataset")
    parser.add_argument("--output_dir", metavar="", type=str, action="store", default=".\\lfw_preprocessed\\", dest="output_dir", help="directory of the preprocessed dataset")

    args = parser.parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    main(args.input_dir, args.output_dir)
