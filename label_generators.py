import os
import cv2
import numpy as np

DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")
FOLDERS = ["Left", "Forward", "Right"]

# coontributes by Tshering Norbu, instead of setting the same steering angle,
# it generates steering angle based on the image and path_name
def calculate_steering_angle(image_path, folder_bias):
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    # convert to hsv and filter for white/yellow lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])  # high value for bright lines
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # edge detection
    edges = cv2.Canny(mask, 50, 150)

    # crop the bottom half
    height, width = edges.shape
    polygon = np.array(
        [[(0, height), (width, height), (width, height // 2), (0, height // 2)]],
        np.int32,
    )
    mask_poly = np.zeros_like(edges)
    cv2.fillPoly(mask_poly, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask_poly)

    # hough transform to find lines
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, 15, minLineLength=20, maxLineGap=10
    )

    calculated_angle = 0.0

    if lines is not None:
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # aboid division by zero
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)

        # if average slope is negative, we are likely leaning left
        if len(slopes) > 0:
            avg_slope = np.mean(slopes)
            # heuristic conversion from slope to steering angle
            calculated_angle = np.clip(avg_slope * 0.5, -1.0, 1.0)

    # blend calculated angle with folder bias
    # if computer vision fails (0.0), it falls back to the folder bias
    # if CV works, it refines the folder bias
    final_angle = (calculated_angle * 0.3) + (folder_bias * 0.7)

    return round(final_angle, 4)


print(f"Generating labels to {CSV_PATH}...")

# define standard bias for folders
folder_map = {"Left": -0.3, "Forward": 0.0, "Right": 0.3}

with open(CSV_PATH, "w") as f:
    # Header
    f.write("image_path,steering_angle\n")

    count = 0
    for folder_name, bias in folder_map.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(folder_path, file)

                # calculates the steering angle
                angle = calculate_steering_angle(full_path, bias)

                # write to csv
                # we store the relative path for portability
                rel_path = os.path.join(folder_name, file)
                f.write(f"{rel_path},{angle}\n")
                count += 1

print(f"Success! Generated labels for {count} images.")
print("Open 'data/driving_log.csv' to inspect values.")
