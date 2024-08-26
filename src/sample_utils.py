import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time

def plot_image_grid(images, n_rows, m_cols, labels=None, figsize=(15, 15), save_path=None):
    """
    Plot a grid of PIL images with optional labels for each row and no whitespace between rows.
    
    Parameters:
    - images: List of PIL images to plot
    - n_rows: Number of rows in the grid
    - m_cols: Number of columns in the grid
    - labels: Optional list of labels for each row (default: None)
    - figsize: Size of the figure (width, height) in inches
    - save_path: Path to save the figure (default: None, which means the figure is not saved)
    """
    fig, axes = plt.subplots(n_rows, m_cols, figsize=figsize)
    
    # Remove all spacing between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    # Ensure axes is always 2D, even for a single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            img_index = i * m_cols + j
            if img_index < len(images):
                # Plot PIL image directly
                ax.imshow(images[img_index])
                ax.axis('off')
            else:
                ax.axis('off')  # Turn off axis for empty subplots
        
        # Add label to the left of each row if labels are provided
        if labels and i < len(labels):
            fig.text(0.01, (n_rows - i - 0.5) / n_rows, labels[i], 
                     va='center', ha='left', fontsize=12, fontweight='bold')

    # Remove any remaining whitespace around the entire plot
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
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