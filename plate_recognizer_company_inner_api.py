from plate_recognizer_company.backend_api import *
import PySimpleGUI as sg
import json
from PIL import Image
import numpy as np
from PIL import Image
import json

def remove_last_char_from_plate(data):
    # Check if the input data is in the expected format
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], dict) and 'type' in data[0] and 'props' in data[0] and 'plate' in data[0]['props']:
        # Get the list of plate values
        plate_values = data[0]['props']['plate']

        # Iterate through the plate values and remove the last character from each
        modified_plate_values = []
        for plate_value_data in plate_values:
            plate_value = plate_value_data['value']
            if plate_value:
                modified_plate_value = plate_value[:-1]
                plate_value_data['value'] = modified_plate_value
                modified_plate_values.append(plate_value_data)

        # Update the plate values in the input data
        data[0]['props']['plate'] = modified_plate_values

        # Return the modified data along with the bounding box
        return data

    # If the input data is not in the expected format, return None
    return None


def crop_license_plate(data, image_path):

    # Extract bounding box for the license plate
    bbox = data[0]["box"]
    xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]

    # Open the image and crop
    img = Image.open(image_path)
    cropped_img = img.crop((xmin, ymin, xmax, ymax))

    # Convert the cropped image to a NumPy array
    cropped_np = np.array(cropped_img)

    return cropped_np


def get_total_vehicle(source) -> int:
    return len(get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Vehicle,
                                                        desired_info=[OptionalInfo.Nothing]))


def get_total_plats(source) -> int:
    return len(get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                        desired_info=[OptionalInfo.Nothing]))


def get_vehicles_coordinates_with_best_plate_text(source) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle)
    if not result:
        return []
    new_result = []
    final_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))

    for itm in new_result:
        if itm[0] is not None:
            croped_image= crop_license_plate(itm,source)
            if is_israeli_license(croped_image):
                final_result.append(itm[0])
            else:
                itm_without_last = remove_last_char_from_plate(itm)
                final_result.append(itm_without_last[0])

    return final_result


def get_plate_coordinates_and_confidence_and_class_name_of_object(source):
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate,
                                                                    OptionalInfo.PlateDetectionScore],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate)
    new_result = []
    if not result:
        return []
    for itm in result:
        coordinates = list(itm[1])
        best_plate = itm[0]["BestPlate"]
        confidence = float(itm[0]["PlateDetectionScore"])
        new_result.append((coordinates, confidence, best_plate))
    # print("new_result", new_result)
    return new_result


def get_vehicle_coordinates_and_confidence_and_class_name(source):
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.VehicleType,
                                                                    OptionalInfo.VehicleTypeDetectionScore],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle)
    new_result = []
    if result == []:
        return []
    for itm in result:
        coordinates = list(itm[1])
        vehicle_type = itm[0]["VehicleType"]
        confidence = float(itm[0]["VehicleTypeDetectionScore"])
        new_result.append((coordinates, confidence, vehicle_type))
    # print("get_vehicle_coordinates_and_confidence_and_class_name", new_result)
    return new_result


def get_plate_coordinates_with_plates_text(source) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate)
    new_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))
    return new_result


def get_vehicles_coordinates(source) -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Vehicle,
                                                    desired_info=[OptionalInfo.Nothing])


def get_plates_coordinates(source) -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                    desired_info=[OptionalInfo.Nothing])


def get_plates_coordinates_with_best_plate(source) -> List[Tuple]:
    result = get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                      desired_info=[OptionalInfo.BestPlate])
    if result is False:
        return False
    new_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))
    return new_result


def get_vehicle_coordinates_by_terms(source,
                                     vehicle_type: OptionalVehicleType = None,
                                     min_vehicle_type_detection_score=None,
                                     region_license_plate_code: OptionalRegionLicense = None,
                                     min_region_license_plate_score=None,
                                     min_plate_detection_score=None,
                                     min_plate_text_reading_score=None, ) \
        -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source,
                                                    desired_info=[OptionalInfo.Nothing],
                                                    plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle,
                                                    vehicle_type=vehicle_type,
                                                    min_vehicle_type_detection_score=min_vehicle_type_detection_score,
                                                    region_license_plate_code=region_license_plate_code,
                                                    min_region_license_plate_score=min_region_license_plate_score,
                                                    min_plate_detection_score=min_plate_detection_score,
                                                    min_plate_text_reading_score=min_plate_text_reading_score)


def get_plate_coordinates_by_terms(source,
                                   vehicle_type: OptionalVehicleType = None,
                                   min_vehicle_type_detection_score=None,
                                   region_license_plate_code: OptionalRegionLicense = None,
                                   min_region_license_plate_score=None,
                                   min_plate_detection_score=None,
                                   min_plate_text_reading_score=None, ) \
        -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source,
                                                    desired_info=[OptionalInfo.Nothing],
                                                    plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate,
                                                    vehicle_type=vehicle_type,
                                                    min_vehicle_type_detection_score=min_vehicle_type_detection_score,
                                                    region_license_plate_code=region_license_plate_code,
                                                    min_region_license_plate_score=min_region_license_plate_score,
                                                    min_plate_detection_score=min_plate_detection_score,
                                                    min_plate_text_reading_score=min_plate_text_reading_score)

def get_vehicle_coordinates_by_term_ui(source,
                                       min_plate_detection_score=None,
                                       color=None,
                                       vehicle_orientation=None,
                                       vehicle_company=None,
                                       vehicle_model=None,
                                       vehicle_plate_number=None) \
        -> List[Tuple]:

    result = get_object_info_and_coordinates_by_terms(source,
                                                    desired_info=[OptionalInfo.BestPlate],
                                                    plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle,
                                                    min_plate_detection_score=min_plate_detection_score,
                                                    color=color,
                                                    vehicle_orientation=vehicle_orientation,
                                                    vehicle_company=vehicle_company,
                                                    vehicle_model=vehicle_model,
                                                    vehicle_plate_number=vehicle_plate_number)

    if not result:
        return []
    new_result = []
    final_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))

    for itm in new_result:
        if itm[0] is not None:
            croped_image= crop_license_plate(itm,source)
            if is_israeli_license(croped_image):
                final_result.append(itm[0])
            else:
                itm_without_last = remove_last_char_from_plate(itm)
                final_result.append(itm_without_last[0])

    return final_result

def get_car_details():
    # Set a theme for the GUI
    sg.theme('DarkAmber')

    # Define the window's layout with added design elements
    layout = [
        [sg.Text('Car Color:', size=(15, 1), font=("Helvetica", 12)), sg.InputText(key='CarColor', font=("Helvetica", 12))],
        [sg.Text('Car Company:', size=(15, 1), font=("Helvetica", 12)), sg.InputText(key='CarCompany', font=("Helvetica", 12))],
        [sg.Text('Car Model:', size=(15, 1), font=("Helvetica", 12)), sg.InputText(key='CarModel', font=("Helvetica", 12))],
        [sg.Text('Car Plate Number:', size=(15, 1), font=("Helvetica", 12)), sg.InputText(key='CarPlateNumber', font=("Helvetica", 12))],
        [sg.Text('Car Orientation:', size=(15, 1), font=("Helvetica", 12)), sg.InputText(key='CarOrientation', font=("Helvetica", 12))],
        [sg.Button('Submit', button_color=('white', 'green'), size=(10, 1)), sg.Button('Exit', button_color=('white', 'red'), size=(10, 1))]
    ]

    # Create the window
    window = sg.Window('Car Details Input', layout)

    car_details = None

    while True:
        event, values = window.read()

        # If user closes window or clicks Exit
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        if event == 'Submit':
            # Filter out keys with empty values
            car_details = {k: v for k, v in values.items() if v}
            break

    window.close()
    return car_details

def get_vehicle_coordinates_by_dict(dict):
    details = dict
    if details:
        get_vehicle_coordinates_by_term_ui(**details)

def is_israeli_license(license_crop_np):
    color = dominant_hue_color(license_crop_np)
    print(color)
    closet_color = closest_lab_shade(color)
    print(closet_color)

    if closet_color != "yellow":
        return False
    return True
import numpy as np
from PIL import Image
from collections import Counter

def most_common_color(img_path):
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        for color, count in sorted_colors:
            if color != (0, 0, 0):
                return color
    return None

def rgb_to_xyz(rgb):
    r, g, b = [x/255.0 for x in rgb]
    r = (r / 12.92) if (r <= 0.04045) else ((r + 0.055) / 1.055) ** 2.4
    g = (g / 12.92) if (g <= 0.04045) else ((g + 0.055) / 1.055) ** 2.4
    b = (b / 12.92) if (b <= 0.04045) else ((b + 0.055) / 1.055) ** 2.4
    r = r * 100.0
    g = g * 100.0
    b = b * 100.0
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return (X, Y, Z)

def xyz_to_lab(xyz):
    X, Y, Z = xyz
    X_ref, Y_ref, Z_ref = 95.047, 100.000, 108.883
    X /= X_ref
    Y /= Y_ref
    Z /= Z_ref
    def f(t):
        if t > (6/29)**3:
            return t**(1/3)
        else:
            return (1/3) * ((29/6)**2) * t + 4/29
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return (L, a, b)

def lab_distance(color1, color2):
    L1, a1, b1 = color1
    L2, a2, b2 = color2
    return np.sqrt((L2 - L1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

def generate_lab_shades(base_rgb, variance=30):
    base_lab = xyz_to_lab(rgb_to_xyz(base_rgb))
    L, a, b = base_lab
    shades = []
    for i in range(-variance, variance+1, 5):
        new_L = min(max(L + i, 0), 100)
        shades.append((new_L, a, b))
    return shades

def closest_lab_shade(input_rgb):
    standard_colors_rgb = {
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255)
    }
    all_shades_lab = {}
    for color_name, base_rgb in standard_colors_rgb.items():
        all_shades_lab[color_name] = generate_lab_shades(base_rgb)
    input_lab = xyz_to_lab(rgb_to_xyz(input_rgb))
    min_distance = float('inf')
    closest_color_name = None
    for color_name, shades in all_shades_lab.items():
        for shade in shades:
            distance = lab_distance(input_lab, shade)
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name
    return closest_color_name

def rgb_to_hsv(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v

def hsv_to_rgb(h, s, v):
    s /= 100.0
    v /= 100.0
    h60 = h / 60.0
    h60f = np.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255.0), int(g * 255.0), int(b * 255.0)
    return r, g, b

def dominant_hue_color(img_array):
    img = Image.fromarray(img_array)
    img = img.convert('RGB')
    pixels = list(img.getdata())
    hues = [rgb_to_hsv(color)[0] for color in pixels if rgb_to_hsv(color)[1] > 20 and rgb_to_hsv(color)[2] > 20]
    hue_counts = Counter(hues)
    dominant_hue = hue_counts.most_common(1)[0][0]
    dominant_rgb = hsv_to_rgb(dominant_hue, 100, 100)
    return dominant_rgb


# img = "/home/nehoray/PycharmProjects/Shaback/Pakmaz/pictures_good_2/001.png"
# print(get_vehicles_coordinates_with_best_plate_text(img))

img2="/home/nehoray/PycharmProjects/Shaback/Pakmaz/pictures_good_2/001.png"
print(get_vehicle_coordinates_by_term_ui(img2, min_plate_detection_score=0.8))
# # print(1)