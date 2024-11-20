# Take in a .mp4 video and output a folder of images 
# Image dimensions: 256×256 pixels

import cv2
import os

video_path = 'videos/horse_square.mp4'
output_dir = 'keyframes/animation_test/'
image_size = (256, 256)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, image_size)
    cv2.imwrite(output_dir + 'frame_%d.png' % frame_count, resized_frame)
    frame_count += 1

cap.release()
print('Extracted %d frames from %s' % (frame_count, video_path))