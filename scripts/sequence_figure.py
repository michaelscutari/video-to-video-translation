import cv2
import matplotlib.pyplot as plt

# Generate a figure of 8 frames of a .mp4 video
# should be a 2 x 8 with frames from two different videos
real_path = 'videos/horse_square.mp4'
naive_generated_path = 'videos/animation.mp4'
better_generated_path = 'videos/animation.mp4' # For now we will use the same video

def load_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    return frames

real_frames = load_video_frames(real_path)
naive_generated_frames = load_video_frames(naive_generated_path)
# better_generated_frames = load_video_frames(better_generated_path)

fig, axes = plt.subplots(3, 8, figsize=(12, 5))

# Add titles for each row
axes[0, 0].set_title('Ground Truth', loc='left')
axes[1, 0].set_title('Naively Generated', loc='left')
axes[2, 0].set_title('Generated with Temporal Continuity', loc='left')

for i in range(8):
    axes[0, i].imshow(real_frames[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(naive_generated_frames[i])
    axes[1, i].axis('off')
    axes[2, i].imshow(naive_generated_frames[i])
    axes[2, i].axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3)
plt.show()
