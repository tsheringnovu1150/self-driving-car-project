import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

# config
DATA_DIR = "./data"
PLOTS_DIR = os.path.join(DATA_DIR, "plots") # dir to save all generated plots
CSV_FILE = os.path.join(DATA_DIR, "driving_log.csv")


def save_figure(fig_name):
    # saves figure to the plots dir
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    plt.savefig(os.path.join(PLOTS_DIR, fig_name), bbox_inches='tight')
    plt.close()

def plot_steering_distribution(angles):
    # plots histogram of steering angles and saves it
    plt.figure(figsize=(10, 5))
    plt.hist(angles, bins=25, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)

    plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label="Center (0.0)")
    plt.legend()

    save_figure("steering_distribution.png")

def visualize_augmentation(image_path, angle, filename):
    # original image vs 3 random augmentations
    plt.figure(figsize=(15, 5))

    # original img
    img_original = utils.load_image(image_path)
    plt.subplot(1, 4, 1)
    plt.imshow(img_original)
    plt.title(f"Original\nAngle: {angle}")
    plt.axis("off")

    # 3 augmented versions
    for i in range(3):
        aug_img, aug_angle = utils.augment(image_path, angle)

        plt.subplot(1, 4, i + 2)
        plt.imshow(aug_img)
        plt.title(f"Augmented {i+1}\nAngle: {aug_angle:.3f}")
        plt.axis("off")

    plt.suptitle("Augmentation Verification (Shadows, Brightness, Translations)")

    save_figure(f"augmentation_{os.path.basename(filename)}.png")


def visualize_preprocessing(image_path, filename):
    # transform from raw img to nn input
    img = utils.load_image(image_path)

    # run preprocessing pipeline
    processed_img = utils.preprocess(img)

    plt.figure(figsize=(10, 5))

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original RGB\n{img.shape}")
    plt.axis("off")

    # preprocess
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img)
    plt.title(f"Network Input (YUV + Crop + Resize)\n{processed_img.shape}")
    plt.axis("off")

    plt.suptitle("Preprocessing Pipeline Check")

    save_figure(f"preprocessing_{os.path.basename(filename)}.png")


def main():
    print("Starting data exploration...")

    # loads the csv
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please check DATA_DIR and CSV_FILE path.")
        return

    df = pd.read_csv(CSV_FILE, header=None, names=['path', 'angle'])

    print(f"Total samples found: {len(df)}")

    # plot distribution
    print("Plotting Steering Distribution...")
    plot_steering_distribution(df['angle'].values)

    # pick a random sample for visual checks
    random_index = np.random.randint(len(df))
    row = df.iloc[random_index]

    # path extraction
    raw_path = str(row['path'])
    filename = raw_path.split('\\')[-1].split('/')[-1]

    if 'left' in raw_path.lower():
        subfolder = "Left"
    elif 'right' in raw_path.lower():
        subfolder = "Right"
    else:
        subfolder = "Center"

    full_path = os.path.join(DATA_DIR, subfolder, filename)
    angle = float(row['angle'])

    if os.path.exists(full_path):
        print(f"\nAnalyzing Sample: {filename}")

        visualize_augmentation(full_path, angle, filename)

        visualize_preprocessing(full_path, filename)

        print(f"Exploration complete. Plots saved in the '{PLOTS_DIR}' directory.")
    else:
        print(f"Could not find image at {full_path}. Check your folder structure.")

if __name__ == "__main__":
    main()


# contributed by: Rinchen Wangdi and Tshering Norbu
