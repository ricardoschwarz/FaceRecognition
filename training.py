import sys
import logging
import argparse
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops


class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def load_model(model_filepath):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with tf.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        logging.error('Missing model file. Exiting')
        sys.exit(-1)


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
    parser.add_argument("--log", metavar="", type=str, action="store", default="INFO", dest="log_level", help="level of the logging, example: INFO, ERROR, DEBUG")
    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Logging level: {args.log_level}")
    
    logging.basicConfig(filename="log.txt", level= getattr(logging, args.log_level.upper()))

    dataset = get_dataset(args.input_dir)
    load_model(".\\model\\20180402-114759\\20180402-114759.pb")
    print("finished")
    