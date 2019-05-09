import argparse
import os
import tensorflow as tf 
from tensorflow.python.framework import ops


def get_dataset(input_dir):
    dataset = []
    classes = os.listdir(input_dir)
    classes.sort()
    
    # images = ops.convert_to_tensor(input_dir, dtype=tf.string)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with preprocessed dataset.")
    parser.add_argument("--input_dir", metavar="", type=str, action="store", default=".\\lfw_preprocessed_small\\", dest="input_dir", help="directory of the preprocessed lfw dataset")

    args = parser.parse_args()
    print(f"Input directory: {args.input_dir}")

    get_dataset(args.input_dir)
    