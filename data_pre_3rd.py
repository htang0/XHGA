import os
import scipy.io
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import numpy as np
import random
import itertools
import functools
import re
import json
import pandas as pd

DATASET_DIR = "/home/user/hdd/train_data/HGR"


def preprocess_inertial_data(inertial_data, target_len=180):
    # Apply Butterworth lowpass filter to reduce noise
    b, a = butter(4, 0.05, btype="lowpass")
    inertial_data = filtfilt(b, a, inertial_data, axis=0)

    # Resample to 180 points
    current_len = inertial_data.shape[0]
    if current_len < target_len:
        # Extend shorter signals by cubic spline interpolation
        old_x = np.linspace(0, 1, current_len)
        new_x = np.linspace(0, 1, target_len)
        cs = CubicSpline(old_x, inertial_data, axis=0)
        inertial_data = cs(new_x)
    elif current_len > target_len:
        # Downsample longer signals
        indices = np.linspace(0, current_len - 1, target_len, dtype=int)
        inertial_data = inertial_data[indices]
    return inertial_data


def preprocess_rgb_data(rgb_data, target_len=64):
    indices = np.linspace(0, len(rgb_data) - 1, target_len, dtype=int)
    return rgb_data[indices]


def preprocess_utd_mhad():
    import cv2

    inertial_dir = DATASET_DIR + "/UTD-MHAD/Inertial"
    inertial_files = os.listdir(inertial_dir)
    inertial_files.sort()

    RGB_dir = DATASET_DIR + "/UTD-MHAD/RGB"
    RGB_files = os.listdir(RGB_dir)
    RGB_files.sort()

    inertial_data_all = []
    rgb_data_all = []
    label_all = []

    file_pattern = re.compile(r"^a(\d+)_s(\d+)_.*")
    for inertial_file, RGB_file in zip(inertial_files, RGB_files):
        assert inertial_file[0:8] == RGB_file[0:8]  # tag should be the same
        match = file_pattern.match(inertial_file)
        label_action = int(match.group(1)) - 1  # to 0-based
        label_subject = int(match.group(2)) - 1  # to 0-based

        inertial_data = scipy.io.loadmat(os.path.join(inertial_dir, inertial_file))
        inertial_data = np.array(inertial_data["d_iner"])  # (*, 6)
        inertial_data = preprocess_inertial_data(inertial_data)

        RGB_data = cv2.VideoCapture(os.path.join(RGB_dir, RGB_file))
        frames = []  # (?, 60, 80, 3)
        while RGB_data.isOpened():
            ret, frame = RGB_data.read()
            if not ret:
                break
            frame = frame[90:450, 200:440]
            frame = cv2.resize(frame, (96, 144), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        rgb_data = np.array(frames)
        rgb_data = preprocess_rgb_data(rgb_data, target_len=32)

        label_all.append(np.array([label_action, label_subject]))
        inertial_data_all.append(inertial_data)
        rgb_data_all.append(rgb_data)

    # train test split by class
    label_all_with_idx = list(enumerate(label_all))
    action_all_train_num = {0.2: 162}
    for label_ratio in [0.3, 0.4, 0.5]:
        train_idx = []
        label_all_with_idx.sort(key=lambda x: x[1][0])
        label_all_by_action_class = itertools.groupby(
            label_all_with_idx, lambda x: x[1][0]
        )
        for _, label_with_idx in label_all_by_action_class:
            label_with_idx = list(label_with_idx)
            random.shuffle(label_with_idx)
            num_train = max(1, int(label_ratio * len(label_with_idx)))
            train_idx.extend([i for i, _ in label_with_idx[:num_train]])
        train_idx.sort()
        print("all train idx by action", len(train_idx))
        action_all_train_num[label_ratio] = len(train_idx)
        np.save(
            DATASET_DIR
            + f"/UTD-MHAD/train_idx_{str(int(label_ratio * 100))}_action.npy",
            train_idx,
        )

    for label_ratio in [0.2, 0.3, 0.4, 0.5]:
        train_idx = []
        label_all_with_idx.sort(key=lambda x: x[1][1])
        label_all_by_subject_class = itertools.groupby(
            label_all_with_idx, lambda x: x[1][1]
        )
        for _, label_with_idx in label_all_by_subject_class:
            label_with_idx = list(label_with_idx)
            random.shuffle(label_with_idx)
            num_train = max(1, int(label_ratio * len(label_with_idx)))
            train_idx.append([i for i, _ in label_with_idx[:num_train]])
        cur_all_train_num = functools.reduce(lambda x, y: x + len(y), train_idx, 0)
        # align number of training samples
        ti = 0
        while cur_all_train_num > action_all_train_num[label_ratio]:
            train_idx[ti].pop()
            cur_all_train_num -= 1
            ti = (ti + 1) % len(train_idx)
        train_idx = functools.reduce(lambda x, y: x + y, train_idx, [])
        train_idx.sort()
        assert len(train_idx) == action_all_train_num[label_ratio]
        print("all train idx by subject", len(train_idx))
        np.save(
            DATASET_DIR
            + f"/UTD-MHAD/train_idx_{str(int(label_ratio * 100))}_subject.npy",
            train_idx,
        )

    inertial_data_all = np.array(inertial_data_all)
    inertial_data_all = (
        inertial_data_all - np.mean(inertial_data_all, axis=0)
    ) / np.std(inertial_data_all, axis=0)
    rgb_data_all = np.array(rgb_data_all)
    label_all = np.array(label_all)

    print(inertial_data_all.shape)
    print(rgb_data_all.shape)
    print(label_all.shape)

    np.save(DATASET_DIR + "/UTD-MHAD/inertial_data_all.npy", inertial_data_all)
    np.save(DATASET_DIR + "/UTD-MHAD/rgb_data_all.npy", rgb_data_all)
    np.save(DATASET_DIR + "/UTD-MHAD/label_all.npy", label_all)


def preprocess_wear():
    import cv2

    all_index_0 = json.load(
        open(
            DATASET_DIR + "/wear-release/cav_label/train_finetune/labels_2553.json", "r"
        )
    )["data"]
    all_index_1 = json.load(
        open(
            DATASET_DIR + "/wear-release/cav_label/test_finetune/labels_1136.json", "r"
        )
    )["data"]

    all_index = list(enumerate(all_index_0 + all_index_1))
    n_samples = len(all_index)

    rgb_data_all = []
    inertial_data_all = []
    label_all = []
    for i, ann in all_index:
        subject_id = int(re.match(r"^sbj_(\d+).*", ann["video_id"]).group(1))
        rgb_data = cv2.VideoCapture(DATASET_DIR + f"/wear-release/{ann['frame_path']}")
        frames = []
        while rgb_data.isOpened():
            ret, frame = rgb_data.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 96), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frames = np.array(frames)
        frames = preprocess_rgb_data(frames, target_len=32)
        rgb_data_all.append(frames)

        imu_data = pd.read_csv(
            DATASET_DIR + f"/wear-release/{ann['imu']}", index_col=False
        ).to_numpy()  # 250, 14 for wear; 150, 64 for cmu
        imu_data = preprocess_inertial_data(imu_data[:, 1:13], target_len=300).astype(
            np.float32
        )
        inertial_data_all.append(imu_data)

        label_all.append(np.array([int(ann["label"]), subject_id]))

    np.save(
        DATASET_DIR + "/wear-release/inertial_data_all.npy",
        np.array(inertial_data_all),
        allow_pickle=False,
    )
    np.save(
        DATASET_DIR + "/wear-release/rgb_data_all.npy",
        np.array(rgb_data_all),
        allow_pickle=False,
    )
    np.save(
        DATASET_DIR + "/wear-release/label_all.npy",
        np.array(label_all),
        allow_pickle=False,
    )

    for label_ratio in [20, 30, 40, 50, 60]:
        label_all_sort_by_action = list(enumerate(label_all))
        label_all_sort_by_action.sort(key=lambda x: x[1][0])

        train_idx_action = []
        val_idx_action = []
        for _, label_with_idx in itertools.groupby(
            label_all_sort_by_action, lambda x: x[1][0]
        ):
            label_with_idx = list(label_with_idx)
            random.shuffle(label_with_idx)
            num_train = max(1, int(label_ratio * len(label_with_idx) / 100))
            train_idx_action.extend([i for i, _ in label_with_idx[:num_train]])
            val_idx_action.extend([i for i, _ in label_with_idx[num_train:]])
        train_idx_action.sort()

        label_all_sort_by_subject = list(enumerate(label_all))
        label_all_sort_by_subject.sort(key=lambda x: x[1][1])

        train_idx_subject = []
        val_idx_subject = []
        for _, label_with_idx in itertools.groupby(
            label_all_sort_by_subject, lambda x: x[1][1]
        ):
            label_with_idx = list(label_with_idx)
            random.shuffle(label_with_idx)
            num_train = max(1, int(label_ratio * len(label_with_idx) / 100))
            train_idx_subject.extend([i for i, _ in label_with_idx[:num_train]])
            val_idx_subject.extend([i for i, _ in label_with_idx[num_train:]])
        train_idx_subject.sort()

        print(len(train_idx_action), len(train_idx_subject))

        while len(train_idx_action) < len(train_idx_subject):
            train_idx_action.append(val_idx_action.pop())

        while len(train_idx_action) > len(train_idx_subject):
            train_idx_subject.append(val_idx_subject.pop())

        np.save(
            DATASET_DIR + f"/wear-release/train_idx_{str(label_ratio)}_action.npy",
            train_idx_action,
        )
        np.save(
            DATASET_DIR + f"/wear-release/train_idx_{str(label_ratio)}_subject.npy",
            train_idx_subject,
        )


utd_mhad_loaded = {
    "inertial": None,
    "rgb": None,
    "label": None,
}


def load_utd_mhad(label_ratio, objective=None):
    if not objective:
        objective = "action"
    if utd_mhad_loaded["inertial"] is None:
        utd_mhad_loaded["inertial"] = np.load(
            DATASET_DIR + "/UTD-MHAD/inertial_data_all.npy"
        )
    if utd_mhad_loaded["rgb"] is None:
        utd_mhad_loaded["rgb"] = np.load(DATASET_DIR + "/UTD-MHAD/rgb_data_all.npy")
    if utd_mhad_loaded["label"] is None:
        utd_mhad_loaded["label"] = np.load(DATASET_DIR + "/UTD-MHAD/label_all.npy")

    inertial_data_all = utd_mhad_loaded["inertial"]
    rgb_data_all = utd_mhad_loaded["rgb"]
    label_all = utd_mhad_loaded["label"]
    train_idx = np.load(
        DATASET_DIR + f"/UTD-MHAD/train_idx_{str(label_ratio)}_{objective}.npy"
    )
    print(inertial_data_all.shape, rgb_data_all.shape, label_all.shape)

    if objective == "action":
        label_idx = 0
    elif objective == "subject":
        label_idx = 1

    inertial_data_train = inertial_data_all[train_idx]
    rgb_data_train = rgb_data_all[train_idx]
    label_train = label_all[train_idx, label_idx]

    val_mask = np.ones(len(inertial_data_all), dtype=bool)
    val_mask[train_idx] = False

    inertial_data_test = inertial_data_all[val_mask]
    rgb_data_test = rgb_data_all[val_mask]
    label_test = label_all[val_mask, label_idx]

    return (
        (inertial_data_all, rgb_data_all, label_all),
        (inertial_data_train, rgb_data_train, label_train),
        (inertial_data_test, rgb_data_test, label_test),
    )


wear_loaded = {
    "inertial": None,
    "rgb": None,
    "label": None,
}


def load_wear(label_ratio, objective=None):
    if not objective:
        objective = "action"
    if wear_loaded["inertial"] is None:
        wear_loaded["inertial"] = np.load(
            DATASET_DIR + "/wear-release/inertial_data_all.npy"
        )
    if wear_loaded["rgb"] is None:
        wear_loaded["rgb"] = np.load(DATASET_DIR + "/wear-release/rgb_data_all.npy")
    if wear_loaded["label"] is None:
        wear_loaded["label"] = np.load(DATASET_DIR + "/wear-release/label_all.npy")

    inertial_data_all = wear_loaded["inertial"]
    rgb_data_all = wear_loaded["rgb"]
    label_all = wear_loaded["label"]
    train_idx = np.load(
        DATASET_DIR + f"/wear-release/train_idx_{str(label_ratio)}_{objective}.npy"
    )
    print(inertial_data_all.shape, rgb_data_all.shape, label_all.shape)

    if objective == "action":
        label_idx = 0
    elif objective == "subject":
        label_idx = 1

    inertial_data_train = inertial_data_all[train_idx]
    rgb_data_train = rgb_data_all[train_idx]
    label_train = label_all[train_idx, label_idx]

    val_mask = np.ones(len(inertial_data_all), dtype=bool)
    val_mask[train_idx] = False

    inertial_data_test = inertial_data_all[val_mask]
    rgb_data_test = rgb_data_all[val_mask]
    label_test = label_all[val_mask, label_idx]

    return (
        (inertial_data_all, rgb_data_all, label_all),
        (inertial_data_train, rgb_data_train, label_train),
        (inertial_data_test, rgb_data_test, label_test),
    )
