import os
import csv
import numpy as np
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
import tensorflow as tf
import pandas as pd

# Input Shape
img_width, img_height, _ = 224, 224, 3

# Load ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

# Add Layer Embedding
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to get embedding
def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

# Directory containing images
images_dir = './hermes_db/output_rect_square'
# Output CSV file
output_csv = './hermes_db/csv/embeddings_n.csv'
output_names_csv = './hermes_db/csv/filenames_n.csv'

# List to store data
data = []
emb = []

# Loop through images
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        img_path = os.path.join(images_dir, filename)
        embedding = get_embedding(img_path)
        data.append(filename)
        emb.append(embedding)

# Save the DataFrame to a CSV file
df = pd.DataFrame(emb)
df.to_csv(output_csv, index=False)
df_name = pd.DataFrame(data)
df_name.to_csv(output_names_csv, index=False)

