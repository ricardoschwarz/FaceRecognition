import argparse
import os
import tensorflow as tf 
from tensorflow.python.framework import ops


class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(input_dir):
    """
    Builds the dataset from the preprocessed image directory.
    """
    dataset = []
    classes = os.listdir(input_dir)
    classes.sort()
    
    for c in classes:
        image_dir = os.path.join(input_dir, c)        
        if os.path.exists(image_dir):
            image_names = os.listdir(image_dir)
            image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]
            dataset.append(ImageClass(c, image_paths))
    return dataset
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with preprocessed dataset.")
    parser.add_argument("--input_dir", metavar="", type=str, action="store", default=".\\lfw_preprocessed_small\\", dest="input_dir", help="directory of the preprocessed lfw dataset")

    args = parser.parse_args()
    print(f"Input directory: {args.input_dir}")

    get_dataset(args.input_dir)
    