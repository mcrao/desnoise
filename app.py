import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import cv2

def denoise_image_svd(noisy_image, k):
    """Denoise image using SVD by keeping top k singular values"""
    # Process each channel separately for color images
    if len(noisy_image.shape) == 3:
        denoised = np.zeros_like(noisy_image, dtype=np.float32)
        for i in range(3):
            U, s, Vt = np.linalg.svd(noisy_image[:,:,i], full_matrices=False)
            # Keep only the k largest singular values
            denoised[:,:,i] = np.dot(U[:, :k], np.dot(np.diag(s[:k]), Vt[:k, :]))
    else:
        U, s, Vt = np.linalg.svd(noisy_image, full_matrices=False)
        denoised = np.dot(U[:, :k], np.dot(np.diag(s[:k]), Vt[:k, :]))
    
    return np.clip(denoised, 0, 255).astype(np.uint8)

def apply_additional_denoising(image):
    """Apply additional denoising techniques"""
    # Convert to float32 for better processing
    float_img = np.float32(image)
    
    # Apply Non-local Means Denoising
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return denoised

def enhance_image_quality(image):
    """Enhance image quality"""
    # Convert to Lab color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl,a,b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        # Apply CLAHE to grayscale image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
    
    return enhanced

def main():
    st.title("Image Quality Enhancement using SVD")
    st.write("""
    Upload a noisy or low-quality image, and this app will enhance it using SVD-based denoising 
    combined with advanced image processing techniques.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload a noisy image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Display original noisy image
        st.subheader("Uploaded Noisy Image")
        st.image(image, use_column_width=True)

        # Add controls
        st.sidebar.subheader("Enhancement Controls")
        k = st.sidebar.slider("SVD Components (k)", 
                            min_value=1, 
                            max_value=min(min(img_array.shape[0], img_array.shape[1]), 200),
                            value=50,
                            help="Adjust to balance noise reduction and detail preservation")
        
        apply_additional = st.sidebar.checkbox("Apply Additional Denoising", value=True)
        enhance_contrast = st.sidebar.checkbox("Enhance Contrast", value=True)

        if st.sidebar.button("Enhance Image"):
            with st.spinner("Enhancing image quality..."):
                # Step 1: SVD Denoising
                denoised = denoise_image_svd(img_array, k)
                
                # Step 2: Additional Denoising (if selected)
                if apply_additional:
                    denoised = apply_additional_denoising(denoised)
                
                # Step 3: Contrast Enhancement (if selected)
                if enhance_contrast:
                    final_image = enhance_image_quality(denoised)
                else:
                    final_image = denoised

                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Noisy Image")
                    st.image(img_array, use_column_width=True)
                
                with col2:
                    st.subheader("Enhanced Image")
                    st.image(final_image, use_column_width=True)

                # Add download button for enhanced image
                enhanced_pil = Image.fromarray(final_image)
                buf = io.BytesIO()
                enhanced_pil.save(buf, format="PNG")
                buf.seek(0)
                
                st.download_button(
                    label="Download Enhanced Image",
                    data=buf,
                    file_name="enhanced_image.png",
                    mime="image/png"
                )

                # Add explanation of what was done
                st.subheader("Enhancement Process:")
                steps = []
                steps.append(f"1. Applied SVD-based denoising with {k} components")
                if apply_additional:
                    steps.append("2. Applied additional noise reduction using Non-local Means Denoising")
                if enhance_contrast:
                    steps.append("3. Enhanced image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)")
                
                for step in steps:
                    st.write(step)

if __name__ == "__main__":
    main()