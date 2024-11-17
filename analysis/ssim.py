# Implement the SSIM metric for comparing two images
# Read both image files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# SSIM meaning
# 1 is perfect similarity
# 0 is no similarity

# Three images are real_pair (real_in, real_out) and fake_out
image_real_out_path = 'data/frame_0_real.png'
image_fake_out_path = 'data/frame_0_fake.png'

# Read the images
image_real_out = cv2.imread(image_real_out_path)
image_fake_out = cv2.imread(image_fake_out_path)

# start out using grayscale images
image_real_out_gray = cv2.cvtColor(image_real_out, cv2.COLOR_BGR2GRAY)
image_fake_out_gray = cv2.cvtColor(image_fake_out, cv2.COLOR_BGR2GRAY)

# Window size
image_size = (256, 256)
window_size = (256, 256)

slide_x = 0
slide_y = 0

window_real = image_real_out_gray[slide_x:slide_x + window_size[0], slide_y: slide_y + window_size[1]]
window_fake = image_fake_out_gray[slide_x:slide_x + window_size[0], slide_y: slide_y + window_size[1]]

# mu_real is the pixel sample mean of image_real_out_grey
mu_real = window_real.mean()
mu_fake = window_fake.mean()

# compute variances
sigma_squared_real = window_real.var()
sigma_squared_fake = window_fake.var()

# compute covariance
covariance = np.cov(window_real.flatten(), window_fake.flatten())[0][1]

# L is the dynamic range of the pixel values
L = 255
k_1 = 0.01
k_2 = 0.03

# computing c_1 and c_2
c_1 = (k_1 * L) ** 2
c_2 = (k_2 * L) ** 2
c_3 = c_2 / 2

luminance = (2 * mu_real * mu_fake + c_1) / (mu_real ** 2 + mu_fake ** 2 + c_1)
contrast = (2 * np.sqrt(sigma_squared_real) * np.sqrt(sigma_squared_fake) + c_2) / (sigma_squared_real + sigma_squared_fake + c_2)
structure = (covariance + c_3) / (np.sqrt(sigma_squared_real) * np.sqrt(sigma_squared_fake) + c_3)

ALPHA = 1
BETA = 1
GAMMA = 1

ssim = luminance ** ALPHA * contrast ** BETA * structure ** GAMMA

# Plot the images side by side
plt.subplot(1, 2, 1)
plt.imshow(window_real, cmap='gray')
plt.title('Real')
plt.subplot(1, 2, 2)
plt.imshow(window_fake, cmap='gray')
plt.title('Fake')

# add to the plot the SSIM value along with the luminance, contrast, and structure values
plt.suptitle('SSIM: %.2f, Luminance: %.2f, Contrast: %.2f, Structure: %.2f' % (ssim, luminance, contrast, structure))

plt.show()