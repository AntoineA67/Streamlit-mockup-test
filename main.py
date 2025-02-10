import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("Laptop Sleeve Logo Mockup Generator")

# =============================================================================
# 1. Load and display the base product image (laptop sleeve)
# =============================================================================

# Make sure you have a file named "laptop_sleeve.png" in your project folder.
base_img = cv2.imread("laptop_sleeve.png")
if base_img is None:
    st.error("Base image not found. Please ensure 'laptop_sleeve.png' exists in your directory.")
    st.stop()

# Convert from BGR (OpenCV default) to RGB for display in Streamlit.
base_img_disp = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
st.image(base_img_disp, caption="Laptop Sleeve Base Image", use_column_width=True)

# =============================================================================
# 2. Set up controls for adjusting the perspective and lighting
# =============================================================================

st.header("Customize Logo Placement and Lighting")

# Get base image dimensions
img_height, img_width = base_img.shape[:2]

st.subheader("Perspective Transformation Coordinates")
st.write("Define the four corner points (in pixels) on the product where the logo should appear.")

# Default coordinates (adjust these to roughly match your desired logo placement on your base image)
default_tl = [int(img_width * 0.3), int(img_height * 0.3)]
default_tr = [int(img_width * 0.7), int(img_height * 0.3)]
default_br = [int(img_width * 0.7), int(img_height * 0.7)]
default_bl = [int(img_width * 0.3), int(img_height * 0.7)]

# Let the admin tweak these coordinates manually
col1, col2, col3, col4 = st.columns(4)
with col1:
    tl_x = st.number_input("Top-Left X", min_value=0, max_value=img_width, value=default_tl[0])
    tl_y = st.number_input("Top-Left Y", min_value=0, max_value=img_height, value=default_tl[1])
with col2:
    tr_x = st.number_input("Top-Right X", min_value=0, max_value=img_width, value=default_tr[0])
    tr_y = st.number_input("Top-Right Y", min_value=0, max_value=img_height, value=default_tr[1])
with col3:
    br_x = st.number_input("Bottom-Right X", min_value=0, max_value=img_width, value=default_br[0])
    br_y = st.number_input("Bottom-Right Y", min_value=0, max_value=img_height, value=default_br[1])
with col4:
    bl_x = st.number_input("Bottom-Left X", min_value=0, max_value=img_width, value=default_bl[0])
    bl_y = st.number_input("Bottom-Left Y", min_value=0, max_value=img_height, value=default_bl[1])

# Assemble the destination quadrilateral (in the coordinate system of the base image)
dst_pts = np.float32([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])

st.subheader("Lighting Adjustments")
brightness = st.slider("Brightness (multiplier)", 0.5, 1.5, 1.0, 0.1)
contrast   = st.slider("Contrast (multiplier)", 0.5, 1.5, 1.0, 0.1)

# =============================================================================
# 3. Upload the company logo
# =============================================================================

st.header("Upload Your Company Logo")
uploaded_file = st.file_uploader("Choose a logo image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded logo image and ensure it has an alpha channel (transparency)
    logo_pil = Image.open(uploaded_file).convert("RGBA")
    logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGBA2BGRA)
    st.image(logo_pil, caption="Uploaded Logo", use_column_width=True)
    
    # =============================================================================
    # 4. Compute the perspective transform and apply lighting adjustments
    # =============================================================================
    
    # Get dimensions of the logo image
    logo_h, logo_w = logo.shape[:2]
    
    # Define source points from the logo (the four corners of the logo image)
    src_pts = np.float32([[0, 0], [logo_w, 0], [logo_w, logo_h], [0, logo_h]])
    
    # Calculate the perspective transform matrix that maps the logo corners to the destination points
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp the logo image using the transformation matrix. We set the output size to the base image size
    # so that the warped logo is positioned correctly.
    warped_logo = cv2.warpPerspective(logo, M, (img_width, img_height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_TRANSPARENT)
    
    # =============================================================================
    # 5. Adjust brightness and contrast on the warped logo
    # =============================================================================
    
    # Separate the color channels (BGR) from the alpha channel
    b_channel, g_channel, r_channel, a_channel = cv2.split(warped_logo)
    warped_logo_bgr = cv2.merge([b_channel, g_channel, r_channel])
    
    # Apply brightness and contrast adjustments. The function applies:
    #    new_pixel = contrast * pixel + beta
    # Here beta is computed from the brightness slider.
    beta = int((brightness - 1) * 255)
    adjusted_logo = cv2.convertScaleAbs(warped_logo_bgr, alpha=contrast, beta=beta)
    
    # Merge back the adjusted color channels with the original alpha channel.
    warped_logo_adjusted = cv2.merge([adjusted_logo, a_channel])
    
    # =============================================================================
    # 6. Composite the warped logo onto the base product image
    # =============================================================================
    
    # Convert the base image to 4 channels (BGRA) to prepare for alpha blending.
    base_img_bgra = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)
    
    # Create an alpha mask from the logo’s alpha channel and normalize it to [0,1]
    alpha_mask = warped_logo_adjusted[:, :, 3].astype(float) / 255.0
    alpha_mask = alpha_mask[:, :, np.newaxis]  # shape: (H, W, 1)
    
    # Extract the color part of the warped logo
    warped_logo_color = warped_logo_adjusted[:, :, :3].astype(float)
    base_img_float = base_img.astype(float)
    
    # Composite the images: for each pixel, the result is:
    #    result = logo_pixel * alpha + base_pixel * (1 - alpha)
    composite = base_img_float * (1 - alpha_mask) + warped_logo_color * alpha_mask
    composite = composite.astype(np.uint8)
    
    # =============================================================================
    # 7. Display and allow download of the final mockup
    # =============================================================================
    
    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    st.image(composite_rgb, caption="Mockup Preview", use_column_width=True)
    
    # Create a download button for the final image
    final_image = Image.fromarray(composite_rgb)
    buf = io.BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Mockup", data=byte_im,
                       file_name="mockup.png", mime="image/png")
