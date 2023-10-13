import json
import os
import shutil
import cv2

from plate_recognizer_company.backend_api import get_objects_detection_api_call_data_of, send_api_request
from plate_recognizer_company.plate_recognizer_company_inner_api import get_vehicles_coordinates_with_best_plate_text


def process_images_in_directory(base_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop over each file in the base directory
    for filename in os.listdir(base_directory):
        # Check if the file is an image (e.g., ends with .jpg or .png)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(base_directory, filename)

            # Save the image inside a specific directory named after the base name of the input image
            image_output_directory = os.path.join(output_directory, os.path.splitext(filename)[0])
            if not os.path.exists(image_output_directory):
                os.makedirs(image_output_directory)

            # Call the function to get the vehicle details
            vehicle_details = send_api_request(filepath) #ToDo specify the function wittch return by terms
            # Save the output as a JSON file
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_filepath = os.path.join(image_output_directory, json_filename)
            with open(json_filepath, 'w') as json_file:
                json.dump(vehicle_details, json_file, indent=4)



            shutil.copy(filepath, image_output_directory)

import os
import json
from PIL import Image, ImageDraw, ImageFont


def draw_vehicle_boxes(json_path, image_path, output_image_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Load a TTF font with a larger size
    # Ensure you have the 'arial.ttf' font file in the current directory or provide the full path
    font = ImageFont.truetype("/home/nehoray/Documents/Roboto-Bold.ttf", 40)  # Adjust the size (40 here) as needed

    for result in data['results']:
        if result['vehicle'] and result['plate']:
            vehicle_box = result['vehicle']['box']
            draw.rectangle([(vehicle_box['xmin'], vehicle_box['ymin']), (vehicle_box['xmax'], vehicle_box['ymax'])], outline="red", width=2)

            plate_box = result['plate']['box']
            draw.rectangle([(plate_box['xmin'], plate_box['ymin']), (plate_box['xmax'], plate_box['ymax'])], outline="blue", width=2)

            plate_value = result['plate']['props']['plate'][0]['value']
            draw.text((plate_box['xmin'], plate_box['ymin'] - 50), plate_value, font=font, fill="blue")

            if 'color' in result['vehicle']['props']:
                vehicle_color = result['vehicle']['props']['color'][0]['value']
                draw.text((plate_box['xmin'], plate_box['ymax'] + 10), vehicle_color, font=font, fill="green")

    image.save(output_image_path)


def process_directory(base_dir):
    # Loop through each item in the base directory
    for item in os.listdir(base_dir):

        item_path = os.path.join(base_dir, item)
        print(item_path)
        # Check if it's a directory
        if os.path.isdir(item_path):
            image_file = None
            json_file = None

            # Loop through each file in the sub-directory
            for file_name in os.listdir(item_path):
                full_path = os.path.join(item_path, file_name)
                if file_name.endswith('.png') and not file_name.startswith("output_"):
                    image_file = full_path
                elif file_name.endswith('.json'):
                    json_file = full_path

            # If both the image and JSON files are found, process them
            if image_file and json_file:
                output_image_path = os.path.join(item_path, "output_" + os.path.basename(image_file))
                draw_vehicle_boxes(json_file, image_file, output_image_path)

if __name__ == '__main__':
    # base_directory = "/home/nehoray/PycharmProjects/Shaback/Pakmaz/pictures_good_2"
    # output_directory = "/home/nehoray/PycharmProjects/Shaback/Pakmaz/output1"
    # process_images_in_directory(base_directory, output_directory)
    # Example usage:
    process_directory("/home/nehoray/PycharmProjects/Shaback/Pakmaz/output1")

