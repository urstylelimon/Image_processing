import os
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# Define configuration parameters
class MyConfig(Config):
    NAME = "my_config"
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MAX_DIM = 1024  # Adjust this value based on your requirements

# Load the dataset
dataset_dir = "output/histogram_equalization"
image_subdirs = os.listdir(dataset_dir)

# Create a training dataset
train_dataset = utils.Dataset()

for image_subdir in image_subdirs:
    image_dir = os.path.join(dataset_dir, image_subdir)
    for image_filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (MyConfig.IMAGE_MAX_DIM, MyConfig.IMAGE_MAX_DIM))

        class_id = int(image_subdir)
        bbox = utils.generate_bbox(image, class_id)
        masks = utils.generate_mask_for_bbox(image, bbox)

        train_dataset.add_image(image_dir, image_filename, image, class_ids=[class_id], masks=[masks])

# Prepare the training data
train_dataset.prepare()

# Train the Mask R-CNN model
model = modellib.MaskRCNN(config=MyConfig(), model_dir="./")
model.train(train_dataset, train_dataset, epochs=20, layers=["heads"])

# Save the trained model
model.keras_model.save_weights("mask_rcnn_weights.h5")
