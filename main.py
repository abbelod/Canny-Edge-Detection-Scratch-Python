from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading Image in GrayScale
image_path = 'sample2.jpg'
img = io.imread(image_path, as_gray = True)
img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Implementing Convolution 
def conv(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # padding the image
    pad_width = ((Hk // 2, Hk // 2), (Wk // 2, Wk // 2))
    padded_image = np.pad(image, pad_width, mode ='edge')

    # Applying convolution
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(kernel * padded_image[i:i+Hk, j:j+Wk])
    
    return out


# Making Gaussian Blur Kernel
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            x_diff = x - center
            y_diff = y - center
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_diff**2 + y_diff**2) / (2*sigma**2))

    return kernel / np.sum(kernel)


# Canny Edge Detection Functions

# Function to compute horizontal Sobel gradient
def partial_x(img):
    sobel_x = np.array([[-1,0,1], [-2, 0, 2], [-1, 0, 1]])
    return conv(img, sobel_x)

# Function to compute vertical Sobel gradient
def partial_y(img):
    sobel_y = np.array([[-1, -2, -1], [0,0,0], [1, 2, 1]])
    return conv(img, sobel_y)


# Function to compute gradient
def gradient(img):
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.rad2deg(np.arctan2(Gy, Gx)) % 360
    return G, theta

 

def non_maximum_suppression(G, theta):
    H, W = G.shape
    out = np.zeros((H, W))

    # Round gradient direction to nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = theta[i, j]
            if angle == 0:
                neighbors = [G[i, j-1], G[i, j+1]]
            elif angle == 90:
                neighbors = [G[i-1, j], G[i+1, j]]
            elif angle == 135:
                neighbors = [G[i-1, j-1], G[i+1, j+1]]
            else:  # 45 degrees
                neighbors = [G[i-1, j+1], G[i+1, j-1]]

            if G[i, j] >= max(neighbors):
                out[i, j] = G[i, j]

    return out

def double_thresholding(img, high, low):
    strongs_edges = (img >= high)
    weak_edges = (img >= low) & (img < high)
    return strongs_edges, weak_edges

def link_edges(strong_edges, weak_edges):
    H, W = strong_edges.shape
    edges = np.copy(weak_edges)

    for i in range(1, H-1):
        for j in range(1, W-1):
            if weak_edges[i, j] and np.any(strong_edges[i-1:i+2, j-1:j+2]):
                edges[i, j] = True
    
    return edges


def canny(img, kernel_size = 5, sigma=1.4, high = 0.5, low=0.02):
    gaussian = gaussian_kernel(kernel_size, sigma)
    print("debug 109")
    smoothed_image = conv(img, gaussian)
    print("debug 111")

    G, theta = gradient(smoothed_image)
    print("debug 114")

    nms_img = non_maximum_suppression(G, theta)
    print("debug 117")

    strong_edges, weak_edges = double_thresholding(nms_img, high, low)
    print("debug 120")

    return link_edges(strong_edges, weak_edges)


def hough_transform(img):
    H, W = img.shape
    diag_len = int(np.ceil(np.sqrt(H**2 + W**2)))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len + 1)
    thetas = np.deg2rad(np.arange(-90, 90))
    accumulator = np.zeros((len(rhos), len(thetas)))

    # Find all edge pixels
    ys, xs = np.nonzero(img)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas



fig, axs = plt.subplots(1, 3)

# Step 1: Canny Edge Detection
edges = canny(img, kernel_size=5, sigma=1.4, high=5, low=0.2)
print("debug 143")
axs[0].set_title('Original Image')
axs[0].imshow(img)
axs[1].set_title('My Implementation')
axs[1].imshow(edges)


# Getting edges using OpenCV
cv2edges = cv2.Canny(img2, 50, 150)
print("debug 1")
axs[2].set_title('OpenCV')
axs[2].imshow(cv2edges)


plt.show()