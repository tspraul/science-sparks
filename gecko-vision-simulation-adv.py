import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage

def advanced_gecko_vision_simulation():
    # Create a sample scene
    print("Creating sample scene for gecko vision simulation")
    # Create a simple night scene with some objects
    x, y = np.meshgrid(np.linspace(-10, 10, 600), np.linspace(-10, 10, 600))
    
    # Dark background
    img = np.zeros((600, 600, 3))
    
    # Add random stars
    stars = np.random.random((600, 600)) > 0.997
    img[stars, :] = 1.0
    
    # Add a "moon"
    moon_mask = (np.sqrt((x+6)**2 + (y+6)**2) < 1.5)
    img[moon_mask, :] = 0.8
    
    # Add some "insects" (prey)
    for i in range(5):
        # Random positions for insects
        pos_x = np.random.randint(-8, 8)
        pos_y = np.random.randint(-8, 8)
        size = np.random.uniform(0.2, 0.5)
        
        insect_mask = (np.sqrt((x-pos_x)**2 + (y-pos_y)**2) < size)
        # Different colors for different insects
        if i % 3 == 0:  # Greenish insect
            img[insect_mask, 0] = 0.1  # R
            img[insect_mask, 1] = 0.8  # G
            img[insect_mask, 2] = 0.1  # B
        elif i % 3 == 1:  # Blueish insect
            img[insect_mask, 0] = 0.1  # R
            img[insect_mask, 1] = 0.1  # G
            img[insect_mask, 2] = 0.8  # B
        else:  # Reddish insect
            img[insect_mask, 0] = 0.8  # R
            img[insect_mask, 1] = 0.1  # G
            img[insect_mask, 2] = 0.1  # B
    
    # Add some "foliage" (background elements)
    for i in range(10):
        pos_x = np.random.randint(-9, 9)
        pos_y = np.random.randint(-9, 9)
        size = np.random.uniform(0.5, 1.5)
        
        foliage_mask = (np.sqrt((x-pos_x)**2 + (y-pos_y)**2) < size)
        # Dark green for foliage
        img[foliage_mask, 0] = 0.05  # R
        img[foliage_mask, 1] = 0.2   # G
        img[foliage_mask, 2] = 0.05  # B
    
    # Create a figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original scene
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Scene (Human Vision)')
    axes[0, 0].axis('off')
    
    # 1. Basic Gecko Vision - Color Sensitivity
    gecko_img1 = img.copy()
    # Geckos have better sensitivity to blue and green wavelengths
    gecko_img1[:,:,0] *= 0.6  # Reduce red
    gecko_img1[:,:,1] *= 1.3  # Enhance green
    gecko_img1[:,:,2] *= 1.2  # Enhance blue
    gecko_img1 = np.clip(gecko_img1, 0, 1)  # Keep values in valid range
    
    axes[0, 1].imshow(gecko_img1)
    axes[0, 1].set_title('Gecko Color Sensitivity')
    axes[0, 1].axis('off')
    
    # 2. Gecko Night Vision
    gecko_img2 = img.copy()
    # Enhance brightness in low light conditions
    luminance = 0.299 * gecko_img2[:,:,0] + 0.587 * gecko_img2[:,:,1] + 0.114 * gecko_img2[:,:,2]
    # Geckos can see in much lower light than humans
    dark_areas = luminance < 0.4
    gecko_img2[dark_areas] *= 2.0
    # Add slight blur to simulate lower acuity in very low light
    gecko_img2 = ndimage.gaussian_filter(gecko_img2, sigma=0.5)
    gecko_img2 = np.clip(gecko_img2, 0, 1)  # Keep values in valid range
    
    axes[1, 0].imshow(gecko_img2)
    axes[1, 0].set_title('Gecko Night Vision')
    axes[1, 0].axis('off')
    
    # 3. Combined Gecko Vision with Motion Sensitivity
    gecko_img3 = gecko_img1.copy()
    # Enhance edges to simulate motion sensitivity
    # Calculate simple edge detection
    edges = np.zeros_like(gecko_img3)
    for i in range(3):
        edges[:,:,i] = ndimage.sobel(gecko_img3[:,:,i])
    edge_magnitude = np.sqrt(np.sum(edges**2, axis=2))
    edge_mask = edge_magnitude > 0.1
    
    # Highlight edges (simulating motion sensitivity)
    for i in range(3):
        gecko_img3[:,:,i] = np.where(edge_mask, np.minimum(gecko_img3[:,:,i] * 1.5, 1.0), gecko_img3[:,:,i])
    
    # Apply night vision enhancement
    dark_areas = luminance < 0.4
    gecko_img3[dark_areas] *= 1.8
    gecko_img3 = np.clip(gecko_img3, 0, 1)  # Keep values in valid range
    
    axes[1, 1].imshow(gecko_img3)
    axes[1, 1].set_title('Complete Gecko Vision Simulation')
    axes[1, 1].axis('off')
    
    # Add gecko-like pupil overlay to the final simulation
    pupil = Ellipse((gecko_img3.shape[1]//2, gecko_img3.shape[0]//2), 
                    width=gecko_img3.shape[1]*0.85, 
                    height=gecko_img3.shape[0]*0.25, 
                    angle=0, 
                    facecolor='none', 
                    edgecolor='white', 
                    linewidth=2)
    axes[1, 1].add_patch(pupil)
    
    plt.tight_layout()
    plt.suptitle('Advanced Gecko Vision Simulation', fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    return fig

# Run the advanced simulation
fig = advanced_gecko_vision_simulation()
plt.show()