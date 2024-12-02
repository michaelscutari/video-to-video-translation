# Convert a video to a PyTorch dataset
# Flexible to include different resolutions, frame rates, and frame sizes

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset:
    def __init__(self, video_path, transform=None, frame_size=(224, 224), frame_rate=1, resolution=(224, 224)):
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.frame_buffer = []
        self.frame_count = 0

    def __getitem__(self, idx):
        while len(self.frame_buffer) <= idx:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            if self.frame_count % self.frame_rate != 0:
                continue
            frame = cv2.resize(frame, self.resolution)
            frame = cv2.resize(frame, self.frame_size)
            if self.transform:
                frame = self.transform(frame)
            self.frame_buffer.append(frame)
        if idx < len(self.frame_buffer):
            return self.frame_buffer[idx]
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        # Ensure all frames are read
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            if self.frame_count % self.frame_rate != 0:
                continue
            frame = cv2.resize(frame, self.resolution)
            frame = cv2.resize(frame, self.frame_size)
            if self.transform:
                frame = self.transform(frame)
            self.frame_buffer.append(frame)
        return len(self.frame_buffer)

    def __del__(self):
        self.cap.release()

def main():
    video_path = "videos/horse_square.mp4"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VideoDataset(video_path, transform=transform, frame_size=(224, 224), frame_rate=1, resolution=(224, 224))
    print(f"Dataset length: {len(dataset)}")
    print(f"First frame shape: {dataset[0].shape}")

main()