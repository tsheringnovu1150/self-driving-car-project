# Self-Driving Car — Behavioral Cloning (Minimal Dataset Version)

This project trains a lightweight self-driving car model using a very small dataset
(54 images) with Keras + Keras Tuner.

The workflow is simple:

1. Create a virtual environment using **uv**
2. Install dependencies
3. Prepare your dataset in `./data`
4. Generate the training CSV labels
5. Explore the data set
6. Train the model
7. Run the simulator using `drive.py`

---

## 1. Create Virtual Environment (using uv)

```bash
uv venv
source .venv/bin/activate
```

---

## 2. Install Requirements

```bash
uv pip install -r requirements.txt
```

---

## 3. Dataset Structure (Required)

Place your images inside the `./data` directory in the following structure:

```
data/
 ├── Center/
 │     ├── frame1.jpg
 │     ├── frame2.jpg
 │     └── ...
 ├── Left/
 │     ├── frame1.jpg
 │     ├── frame2.jpg
 │     └── ...
 ├── Right/
 │     ├── frame1.jpg
 │     ├── frame2.jpg
 │     └── ...
```

The script will automatically generate a CSV file from these folders
and apply steering correction for left/right cameras.

---

## 4. Generate Labels

This step creates **driving_log.csv**.

Run:

```bash
uv run python label_generators.py
```

You should now see a CSV file created inside `./data`, containing:

- `path`
- `steering_angle`

---

## 5. Explore the data

```bash
uv run python data_explorations.py
```

This will:

- generate plot folder in ./data

---

## 6. Train the Model

```bash
uv run python model.py
```

This will:

- Load your data
- Run Keras Tuner
- Train the final CNN
- Save:

```
model_best.h5
model_final.h5
```

---

## 7. Run the Simulator (Inference)

Use the trained model to control the car:

```bash
uv run python drive.py model_final.h5
```

or

```bash
uv run python drive.py model_best.h5
```
