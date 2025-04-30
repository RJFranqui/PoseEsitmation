import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class SquatPhaseDataset(Dataset):
    def __init__(self, csv_path, seq_length=30, threshold_pct=50, sigma=1.5):
        """
        Sequence-wise dataset using frame-level UP/DOWN labeling
        with smoothing and percentile-based threshold.
        """
        self.seq_length = seq_length
        self.threshold_pct = threshold_pct
        self.sigma = sigma
        self.scaler = StandardScaler()

        self.data, self.labels = self._load_and_preprocess(csv_path)
        self.raw_means = np.mean(self.data, axis=(0, 1))
        self.raw_stds = np.std(self.data, axis=(0, 1))
        self._normalize_data()

    def _calculate_angles(self, df):
        angles = []
        for (sample_id, time), frame in df.groupby(['Sample_ID', 'Time']):
            try:
                joints = {}
                for jid in [11, 12, 23, 24, 25, 26, 27, 28]:
                    joint_data = frame[frame['ID'] == jid]
                    if len(joint_data) == 0:
                        raise ValueError(f"Missing joint {jid}")
                    joints[jid] = joint_data[['World_X','World_Y','World_Z']].values[0]

                left_thigh = joints[23] - joints[25]
                left_shin = joints[27] - joints[25]
                right_thigh = joints[24] - joints[26]
                right_shin = joints[28] - joints[26]
                torso = (joints[11] + joints[12])/2 - (joints[23] + joints[24])/2

                def safe_angle(v1, v2):
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    return np.clip(np.arccos(cos), 0, np.pi)

                angles.append({
                    'time': time,
                    'sample_id': sample_id,
                    'left_knee': safe_angle(left_thigh, left_shin),
                    'right_knee': safe_angle(right_thigh, right_shin),
                    'left_hip': safe_angle(torso, left_thigh),
                    'torso_lean': np.arctan2(torso[1], abs(torso[2]))
                })
            except:
                continue
        return pd.DataFrame(angles)

    def _label_frames(self, angle_df):
        angle_df = angle_df.sort_values(['sample_id', 'time'])

        features = angle_df[['left_knee', 'right_knee', 'left_hip', 'torso_lean']].values
        smoothed = gaussian_filter1d(features, sigma=self.sigma, axis=0)

        mean_knee = (smoothed[:, 0] + smoothed[:, 1]) / 2
        threshold = np.percentile(mean_knee, self.threshold_pct)
        labels = (mean_knee >= threshold).astype(int)

        print(f"Threshold (rad): {threshold:.2f} | (deg): {np.rad2deg(threshold):.1f}")
        print(f"Labeled as UP: {np.sum(labels == 1)}, DOWN: {np.sum(labels == 0)}")

        return smoothed, labels

    def _create_sequences(self, features, labels):
        sequences = []
        seq_labels = []
        for i in range(0, len(features) - self.seq_length + 1, self.seq_length // 3):
            window = features[i:i + self.seq_length]
            window_labels = labels[i:i + self.seq_length]
            if len(window_labels) < self.seq_length:
                continue
            
            # Calculate frame-to-frame differences
            delta = np.diff(window, axis=0, prepend=window[0:1])

            # Concatenate position and motion (velocity) features
            full_features = np.concatenate([window, delta], axis=1)  # Now shape (seq_length, 8)

            majority = np.bincount(window_labels).argmax()
            sequences.append(full_features)
            seq_labels.append(majority)
            
        return np.array(sequences), np.array(seq_labels)

    # def _create_sequences(self, features, labels):
    #     """Create sequences with adaptive labeling for squat phases using features and labels arrays."""
    #     sequences = []
    #     seq_labels = []
        
    #     # features: (num_frames, 4)
    #     # labels:   (num_frames,)
        
    #     # Apply feature-specific smoothing
    #     smoothed = np.array([
    #         gaussian_filter1d(features[:, i], sigma=self.sigma * (1 + i * 0.2))
    #         for i in range(features.shape[1])
    #     ]).T  # shape (num_frames, 4)
        
    #     # Create sequences with 66% overlap
    #     for i in range(0, len(features) - self.seq_length + 1, self.seq_length // 3):
    #         window = smoothed[i:i+self.seq_length]
            
    #         # Calculate knee angles (average of left and right)
    #         knee_angles = (window[:, 0] + window[:, 1]) / 2
            
    #         # Calculate angle differences
    #         angle_diffs = np.diff(knee_angles)
            
    #         # Calculate total knee angle change
    #         total_angle_change = np.abs(knee_angles[-1] - knee_angles[0])
            
    #         # Phase detection logic (adaptive labeling)
    #         if total_angle_change < 0.07:  # Equivalent to self.min_angle_change
    #             current_label = seq_labels[-1] if seq_labels else 0
    #         elif np.all(angle_diffs > 0):
    #             current_label = 1  # UP phase
    #         elif np.all(angle_diffs < 0):
    #             current_label = 0  # DOWN phase
    #         else:
    #             current_label = seq_labels[-1] if seq_labels else 0
            
    #         sequences.append(window)
    #         seq_labels.append(current_label)
        
    #     return np.array(sequences), np.array(seq_labels)

    def _load_and_preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        angle_df = self._calculate_angles(df)
        features, labels = self._label_frames(angle_df)
        return self._create_sequences(features, labels)

    def _normalize_data(self):
        orig_shape = self.data.shape
        flattened = self.data.reshape(-1, orig_shape[-1])
        self.data = self.scaler.fit_transform(flattened).reshape(orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        return pad_sequence(sequences, batch_first=True), torch.cat(labels)

    def analyze(self):
        print(f"\n{' Sequence Dataset Analysis ':=^80}")
        print(f"Total sequences: {len(self.data)}")
        print(f"Class balance: DOWN {sum(self.labels==0)} | UP {sum(self.labels==1)}")
        print(f"Raw means (deg): {np.rad2deg(self.raw_means)}")
        print(f"Raw stds (deg): {np.rad2deg(self.raw_stds)}")

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.hist(self.labels, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
        plt.xticks([0, 1], ['DOWN', 'UP'])
        plt.title('Class Distribution')

        plt.subplot(122)
        for i, name in enumerate(['Left Knee','Right Knee','Left Hip','Torso Lean']):
            plt.hist(np.rad2deg(self.data[:,:,i].ravel()), bins=50, alpha=0.5, label=name)
        plt.title('Angle Feature Distributions')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize dataset
    dataset = SquatPhaseDataset(
        "Squat_Test.csv",
        seq_length=30,
        min_velocity=0.025,
        sigma=2.0
    )
    
    # Analyze dataset
    dataset.analyze()
    
    # Create balanced dataloader
    def create_balanced_loader(dataset, batch_size=32):
        class_weights = 1. / np.bincount(dataset.labels)
        sample_weights = class_weights[dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )
    
    train_loader = create_balanced_loader(dataset)
    batch = next(iter(train_loader))
    print(f"\nBatch shapes - Sequences: {batch[0].shape}, Labels: {batch[1].shape}")