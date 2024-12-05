import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_video_frames(video_path, num_frames=60, crop_height=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[crop_height:, :, :]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    print(f"Loaded {len(frames)} frames")
    return frames

def create_temporal_continuity_image(frames, slice_width=4):
    height, width, _ = frames[0].shape
    print(f"Frame shape: {frames[0].shape}")
    center_x = width // 2
    slices = [frame[center_x - slice_width // 2:center_x + slice_width // 2, :] for frame in frames]
    # rotate slices counter-clockwise 90 degrees
    slices = [np.rot90(slice) for slice in slices]
    #print a slice
    combined_image = np.concatenate(slices, axis=1)
    return combined_image

video_paths = [
    'videos/horse_square.mp4',
    'videos/animation.mp4',
    'videos/animation.mp4'
]

all_frames = [load_video_frames(video_path) for video_path in video_paths]
continuity_images = [create_temporal_continuity_image(frames) for frames in all_frames]

plt.figure(figsize=(20, 10))
for i, continuity_image in enumerate(continuity_images):
    plt.subplot(1, len(continuity_images), i + 1)
    plt.imshow(cv2.cvtColor(continuity_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Temporal Continuity Image for Video {i+1}')
    plt.axis('off')
plt.show()
