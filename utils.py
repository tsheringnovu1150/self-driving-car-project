import cv2
import matplotlib.image as mpimg
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(image_path):
    # reads in RGB format
    return mpimg.imread(image_path)


def crop(image): # Tshering Norbu
    # since the training dataset already zoom and crop
    height = image.shape[0]
    if height > 90:
        return image[60:-25, :, :]
    return image


def resize(image): # Tshering Norbu
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image): # Tshering Norbu
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image): # Tshering Norbu
    image = crop(image)
    image = resize(image)
    # convert to YUV (NVIDIA architecture recommendation)
    image = rgb2yuv(image)
    # blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

# Rinchen Wangdi
def random_translate(image, steering_angle, range_x=100, range_y=10):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

# Rinchen Wangdi
def random_shadow(image):
    # nput must be RGB, puts random shadow on image
    h, w = image.shape[0], image.shape[1]
    x1, y1 = w * np.random.rand(), 0
    x2, y2 = w * np.random.rand(), h
    xm, ym = np.mgrid[0:h, 0:w]

    mask = np.zeros_like(image[:, :, 1])
    mask[((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1)) > 0] = 1
    cond = mask == np.random.randint(2)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2][cond] = hsv[:, :, 2][cond] * np.random.uniform(0.4, 0.7)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Tsheing Norbu
def random_brightness(image):
    # input must be RGB, puts random brightness on image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# since the data set is less, use fliping of right and left image
# Tshering Norbu
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

#  Rinchen Wangi
def augment(path, steering_angle):
    # load RGB
    image = load_image(path)

    # perform augmentations on RGB data
    if np.random.rand() < 0.5:
        image, steering_angle = random_translate(image, steering_angle)
    if np.random.rand() < 0.5:
        image = random_shadow(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)

    image, steering_angle = random_flip(image, steering_angle)

    return image, steering_angle

# Rinchen Wangdi
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

    while True:
        i = 0
        while i < batch_size:
            index = np.random.randint(len(image_paths))
            path = image_paths[index]
            angle = steering_angles[index]

            # rejection sampling for straight driving, bias on left and right
            # since data set center: 30,  left and right: 12 each
            if abs(angle) < 0.05:
                if np.random.rand() > 0.7:
                    continue

            try:
                if is_training:
                    # augment returns RGB image
                    img, angle = augment(path, angle)
                else:
                    img = load_image(path)

                # preprocess converts RGB to YUV and resizes
                images[i] = preprocess(img)
                steers[i] = angle
                i += 1
            except Exception as e:
                continue

        yield images, steers

# since the image have to YUV for nvidia
# so use YUV
