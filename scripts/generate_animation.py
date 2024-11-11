import os
import cv2

# Take the output images and create an animation
keyframes = 'keyframes/animation_test_output/'
output_dir = 'videos/animation_test_output/'

start_frame = 0
end_frame = 59

# the pngs will be called frame_0_fake.png, frame_1_fake.png, ..., frame_59_fake.png
# The animation should be 30 frames per second

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_array = []
for i in range(start_frame, end_frame+1):
    img = cv2.imread(keyframes + 'frame_%d_fake.png' % i)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

# save as a .mp4 file
out = cv2.VideoWriter(output_dir + 'animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for frame in frame_array:
    out.write(frame)

out.release()  # Release the VideoWriter object
print(f"Video saved at {os.path.join(output_dir, 'animation.mp4')}")