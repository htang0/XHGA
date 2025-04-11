import os
import numpy as np
import torch
import random

from data_pre_3rd import preprocess_utd_mhad, preprocess_wear, load_utd_mhad, load_wear

random.seed(0)

DATASET_DIR = "/home/user/hdd/train_data/HGR"


class DualMultimodalDataset:
    def __init__(self, x1, x2, y, x1_1, x2_1, y_1):
        self.data1 = x1
        self.data2 = x2
        self.labels = y
        self.data1_1 = x1_1
        self.data2_1 = x2_1
        self.labels_1 = y_1

    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        sensor_data1 = torch.from_numpy(self.data1[index])
        sensor_data1 = torch.unsqueeze(sensor_data1, 0)

        sensor_data2 = torch.from_numpy(self.data2[index])

        activity_label = torch.from_numpy(np.array(self.labels[index])).long()

        sensor_data1_1 = torch.from_numpy(self.data1_1[index])
        sensor_data1_1 = torch.unsqueeze(sensor_data1_1, 0)

        sensor_data2_1 = torch.from_numpy(self.data2_1[index])

        activity_label_1 = torch.from_numpy(np.array(self.labels_1[index])).long()

        return (
            sensor_data1,
            sensor_data2,
            activity_label,
            sensor_data1_1,
            sensor_data2_1,
            activity_label_1,
        )


class MultimodalDataset:
    def __init__(self, x1, x2, y):
        self.data1 = x1
        self.data2 = x2
        self.labels = y

    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        sensor_data1 = torch.from_numpy(self.data1[index])
        sensor_data1 = torch.unsqueeze(sensor_data1, 0)

        sensor_data2 = torch.from_numpy(self.data2[index])

        activity_label = torch.from_numpy(np.array(self.labels[index])).long()

        return sensor_data1, sensor_data2, activity_label


class MultimodalUnlabeledDataset:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, index):
        sensor_data1 = torch.from_numpy(self.x1[index])
        sensor_data1 = torch.unsqueeze(sensor_data1, 0)

        sensor_data2 = torch.from_numpy(self.x2[index])

        return sensor_data1, sensor_data2


def load_class_data_single(
    sensor_str, activity_class, train_test_flag, label_rate, dataset, class_type
):
    data_all_samples = []
    data_dir = ""
    if train_test_flag == 1:  # Train
        data_dir = (
            f"../data/train/label-rate-{label_rate}-percent/{dataset}/{sensor_str}/"
        )
    elif train_test_flag == 2:  # Test
        data_dir = f"../data/{dataset}/{sensor_str}/"

    if class_type == "s":
        cls_str = f"s{activity_class + 1}_"
    elif class_type == "a":
        cls_str = f"a{activity_class + 1}_"
    else:
        raise ValueError("Invalid class_type")

    files = os.listdir(data_dir)

    for filename in files:
        if cls_str in filename:
            loaded_sample = np.load(os.path.join(data_dir, filename))
            data_all_samples.extend(loaded_sample)

    data_all_samples = np.array(data_all_samples)
    return data_all_samples


def load_unlabel_data(label_rate):
    x1 = []
    x2 = []

    for unlabeled_subdir in ["unlabel", "unlabel-2"]:
        for sensor_type in ["imu_denoise_3inter", "image"]:
            data_directory = f"{DATASET_DIR}/train/label-rate-{label_rate}-percent/{unlabeled_subdir}/{sensor_type}/"
            directory_files = os.listdir(data_directory)
            directory_files.sort()

            for filename in directory_files:
                loaded_sample = np.load(os.path.join(data_directory, filename))
                if sensor_type == "imu_denoise_3inter":
                    x1.extend(loaded_sample)
                elif sensor_type == "image":
                    x2.extend(loaded_sample)

    x1 = np.array(x1).reshape(-1, 300, 6)
    x1 = np.concatenate((x1, x1), axis=-1)

    x2 = np.array(x2).reshape(-1, 32, 72, 128, 3)
    x2 = x2.swapaxes(2, 4).swapaxes(1, 2)

    return x1, x2


def load_data(
    num_of_total_class, num_per_class, train_test_flag, label_rate, dataset, class_type
):
    x1 = []
    x2 = []
    y = []

    for class_id in range(num_of_total_class):
        x1_data = load_class_data_single(
            "imu_denoise_3inter",
            class_id,
            train_test_flag,
            label_rate,
            dataset,
            class_type,
        ).reshape(-1, 300, 6)

        x2_data = load_class_data_single(
            "image", class_id, train_test_flag, label_rate, dataset, class_type
        ).reshape(-1, 32, 72, 128, 3)

        total_samples_in_class = x1_data.shape[0]
        class_labels = np.ones(total_samples_in_class) * class_id

        num_to_sample = min(num_per_class[class_id], total_samples_in_class)
        sampled_indices = random.sample(range(total_samples_in_class), num_to_sample)

        sampled_x1_data = x1_data[sampled_indices]
        sampled_x2_data = x2_data[sampled_indices]
        sampled_labels = class_labels[sampled_indices]

        x1.extend(sampled_x1_data)
        x2.extend(sampled_x2_data)
        y.extend(sampled_labels)

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    x1 = np.concatenate((x1, x1), axis=-1)
    x2 = x2.swapaxes(2, 4).swapaxes(1, 2)

    return x1, x2, y
