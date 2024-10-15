from PIL import Image
import argparse
from glob import glob
import os


def resize_image(input_path, output_path, new_width, new_height):
    # Open an image file
    with Image.open(input_path) as img:
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Save it back to disk
        path = output_path + '/' + "/".join(input_path.split('/')[-2:])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img_resized.save(path)

# Parameters
# Create the argument parser
parser = argparse.ArgumentParser(description='Resize an image')

# Add the input image path argument
parser.add_argument('input_image_path', type=str, help='Path to the input image')

# Add the output image path argument
parser.add_argument('output_image_path', type=str, help='Path to save the output image')

# Parse the command line arguments
args = parser.parse_args()

# Get the input and output image paths from the command line arguments
input_image_path = args.input_image_path
output_image_path = args.output_image_path
os.makedirs(output_image_path, exist_ok=True)
# Parameters
new_width = 1024
new_height = 256

# Resize the image
image_files = glob(f'{input_image_path}/*.png')
for image_file in image_files:
    resize_image(image_file, output_image_path, new_width, new_height)
    print(f"Image {image_file} has been resized to {new_width}x{new_height} and saved to {output_image_path}")
