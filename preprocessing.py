import os
import threading

import cv2
import dlib

shape_predictor_path="shape_predictor_68_face_landmarks.dat"

def align_face(image):
    """
    Aligns a persons face within the image using dlib.
    Returns a 320x320 image of the aligned face.
    """
    face_detector=dlib.get_frontal_face_detector()
    shape_predictor=dlib.shape_predictor(shape_predictor_path)
    detected_faces=face_detector(image, 1)
    num_faces = len(detected_faces)
    if num_faces == 0:
        return False
    faces = dlib.full_object_detections()
    for detection in detected_faces:
        faces.append(shape_predictor(image, detection))
    image = dlib.get_face_chip(image, faces[0])
    return image


def save(image, output_path):
    """
    Saves an image to the output_path.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def preprocess(image_path, output_path):
    """
    Preprocess the image found under image_path and save the processed image to output_path.
    Preprocessing consists of color converting (COLOR_BGR2RGB), croping and saving.
    """
    image = cv2.imread(image_path, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = align_face(image)
    save(image, output_path)
    