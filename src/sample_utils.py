import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import numpy as np
import os
import time


def plot_image_grid(images, n_rows, m_cols, labels=None, save_path=None, img_size=32):
    """
    Plot a grid of PIL images with rotated labels for each row and no whitespace between images.
    
    Parameters:
    - images: List of PIL images to plot
    - n_rows: Number of rows in the grid
    - m_cols: Number of columns in the grid (should be 8 for 8 32x32 images)
    - labels: Optional list of labels for each row (default: None)
    - save_path: Path to save the figure (default: None, which means the figure is not saved)
    """
    dpi = 100  # Set DPI for the figure
    label_width = 0.7  # Width for labels in inches (increased for padding)
    
    # Calculate figure size
    fig_width = (img_size * m_cols) / dpi + label_width
    fig_height = (img_size * n_rows) / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Create grid with space for labels
    gs = gridspec.GridSpec(n_rows, m_cols + 1, width_ratios=[label_width * dpi / img_size] + [1] * m_cols)
    gs.update(wspace=0, hspace=0)  # Remove spacing between images
    
    for i in range(n_rows):
        # Add label
        if labels and i < len(labels):
            ax = fig.add_subplot(gs[i, 0])
            ax.text(0.9, 0.5, labels[i], rotation=0, ha='right', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # Add images
        for j in range(m_cols):
            ax = fig.add_subplot(gs[i, j + 1])
            img_index = i * m_cols + j
            if img_index < len(images):
                ax.imshow(images[img_index])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)  # Close the figure to free up memory


def save_images_as_png(images, destination_folder):
    """
    Save each image in the list as a PNG file with a unique time-based identifier
    in the specified destination folder.
    
    Parameters:
    - images: List of PIL images to save
    - destination_folder: Path to the folder where images will be saved
    
    Returns:
    - List of paths to the saved image files
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Generate a unique identifier based on the current time
    time_id = int(time.time() * 1000)  # milliseconds since epoch
    
    saved_paths = []
    for i, img in enumerate(images):
        file_name = f"image_{time_id}_{i:04d}.png"
        file_path = os.path.join(destination_folder, file_name)
        img.save(file_path, "PNG")
        saved_paths.append(file_path)
        print(f"Saved {file_path}")
    
    return saved_paths