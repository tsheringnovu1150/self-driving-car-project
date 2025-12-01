import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

import utils

# init server and app
sio = socketio.Server()
app = Flask(__name__)

model = None
speed_limit = 10  # max speed target


# event handlers
@sio.on("telemetry")
def telemetry(sid, data):
    """
    Callback function triggered when the simulator sends data.
    Driving Logic
    """
    if data:
        current_speed = float(data["speed"])

        image_data = data["image"]
        image_pil = Image.open(BytesIO(base64.b64decode(image_data)))

        try:
            image_array = np.asarray(image_pil)

            processed_image = utils.preprocess(image_array)
            batch_image = np.array([processed_image])
            steering_angle = float(model.predict(batch_image, batch_size=1))

            global speed_limit
            if current_speed > speed_limit:
                speed_limit = 10
            else:
                speed_limit = 15

            throttle = 1.0 - (steering_angle**2) - (current_speed / speed_limit) ** 2
            throttle = np.clip(throttle, 0.0, 1.0)

            print(
                f"Steer: {steering_angle:.4f} | Throttle: {throttle:.4f} | Speed: {current_speed:.2f}"
            )
            send_control(steering_angle, throttle)

            if args.image_folder != "":
                timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
                image_filename = os.path.join(args.image_folder, timestamp + ".jpg")
                image_pil.save(image_filename)

        except Exception as e:
            print(f"Error in telemetry processing: {e}")
            send_control(0, 0)
    else:
        sio.emit("manual", data={}, skip_sid=True)


@sio.on("connect")
def connect(sid, environ):
    """
    triggered when the simulator connects to the script.
    """
    print("Simulator Connected! (SID: ", sid, ")")
    send_control(0, 0.1)


def send_control(steering_angle, throttle):
    """
    Helper to format and emit the control command
    """
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
        skip_sid=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote Driving Client")
    parser.add_argument(
        "model",
        type=str,
        help="Path to model h5 file. Model should be on the same path.",
    )
    parser.add_argument(
        "image_folder",
        type=str,
        nargs="?",
        default="",
        help="Path to image folder. This is where the images from the run will be saved.",
    )
    args = parser.parse_args()
    print(f"Loading model from {args.model}...")
    custom_objects = {"mse": MeanSquaredError()}

    try:
        model = load_model(args.model, custom_objects=custom_objects, safe_mode=False)
        print("Model loaded successfully")
    except OSError:
        print("Error: Model file not found. Please check the path.")
        exit()
    except Exception as e:
        print(f"Critical error loading model: {e}")
        exit()

    # recording
    if args.image_folder != "":
        print(f"Recording run to folder: {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING ENABLED")
    else:
        print("NOT RECORDING")

    app = socketio.Middleware(sio, app)
    print("Server starting on port 4567...")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)


# both of us contributed on this file and took time as stimulator works only on old packages
# use some code from file shared by prof.
