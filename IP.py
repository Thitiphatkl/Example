import streamlit as st
import numpy as np
import cv2
from skimage import exposure, color
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Image Processing Toolbox", layout="wide")

# -------------------------
# Utility Functions
# -------------------------

def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def normalize_image(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def show_histogram(image, ax, title="Histogram"):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ax.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    ax.set_title(title)
    ax.set_xlim([0, 256])

def display_results(original, processed, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
    axs[0, 1].set_title(f"Processed Image ({title})")
    axs[0, 1].axis("off")

    show_histogram(original, axs[1, 0], "Original Histogram")
    show_histogram(processed, axs[1, 1], "Processed Histogram")

    st.pyplot(fig)

# -------------------------
# Image Processing Methods
# -------------------------

def linear_negative(img):
    return 255 - img

def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def piecewise_linear(img, c1, c2, t1, t2):
    img = img.astype(np.float32)
    result = np.zeros_like(img)
    mask1 = img <= c1
    mask2 = (img > c1) & (img <= c2)
    mask3 = img > c2

    result[mask1] = t1 / c1 * img[mask1]
    result[mask2] = ((t2 - t1) / (c2 - c1)) * (img[mask2] - c1) + t1
    result[mask3] = ((255 - t2) / (255 - c2)) * (img[mask3] - c2) + t2
    return normalize_image(result)

def log_transform(img, c):
    img = img.astype(np.float32)
    img = c * np.log1p(img)
    img = img * (255.0 / img.max())
    return normalize_image(img)

def gamma_transform(img, gamma, c):
    img = img.astype(np.float32) / 255.0
    img = c * np.power(img, gamma)
    img = img * 255.0 / img.max()
    return normalize_image(img)

def histogram_equalization(img):
    if len(img.shape) == 2:
        return exposure.equalize_hist(img) * 255
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def adaptive_hist_eq(img):
    if len(img.shape) == 2:
        return exposure.equalize_adapthist(img) * 255
    else:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab[:, :, 0] = exposure.equalize_adapthist(img_lab[:, :, 0]) * 255
        return cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

def clahe_processing(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

# -------------------------
# Streamlit Interface
# -------------------------

st.title("🖼️ Image Processing Toolbox")
st.write("Upload an image and apply various enhancement techniques.")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.sidebar.header("Processing Options")
    method = st.sidebar.selectbox("Choose a method", [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear Transformation",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization",
        "CLAHE"
    ])

    is_grayscale = st.sidebar.checkbox("Convert to Grayscale", value=False)

    if is_grayscale:
        img_np = to_grayscale(img_np)

    # Handle parameters
    if method == "Piecewise Linear Transformation":
        c1 = st.sidebar.slider("C1", 0, 255, 80)
        c2 = st.sidebar.slider("C2", c1 + 1, 255, 160)
        t1 = st.sidebar.slider("T1", 0, 255, 50)
        t2 = st.sidebar.slider("T2", 0, 255, 200)
    elif method == "Log Transformation":
        c = st.sidebar.slider("Constant c", 1.0, 10.0, 5.0)
    elif method == "Gamma Transformation":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0)
        c = st.sidebar.slider("Constant c", 1.0, 5.0, 1.0)
    elif method == "CLAHE":
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile_size = st.sidebar.slider("Tile Grid Size", 2, 16, 8, step=2)
        tile_grid_size = (tile_size, tile_size)

    # Apply selected method
    if method == "Linear Negative":
        result = linear_negative(img_np)
    elif method == "Contrast Stretching":
        result = contrast_stretching(img_np)
    elif method == "Piecewise Linear Transformation":
        result = piecewise_linear(img_np, c1, c2, t1, t2)
    elif method == "Log Transformation":
        result = log_transform(img_np, c)
    elif method == "Gamma Transformation":
        result = gamma_transform(img_np, gamma, c)
    elif method == "Histogram Equalization":
        result = histogram_equalization(img_np)
    elif method == "Adaptive Histogram Equalization":
        result = adaptive_hist_eq(img_np)
    elif method == "CLAHE":
        result = clahe_processing(img_np, clip_limit, tile_grid_size)

    result = normalize_image(result)

    # Display results
    display_results(img_np, result, method)

else:
    st.info("Please upload an image to get started.")
