import numpy as np
import random
from PIL import Image, ImageDraw
from bones_utils import *

def generate_embedding_images(
                            skeletons,
                            upper_embeddings,
                            lower_embeddings,
                            l_arm_embeddings,
                            r_arm_embeddings,
                            upper_body_indices,
                            lower_body_indices,
                            l_arm_body_indices,
                            r_arm_body_indices,
                            upper_filename="upper_embeddings.png", 
                            lower_filename="lower_embeddings.png", 
                            l_arm_filename="l_arm_embeddings.png", 
                            r_arm_filename="r_arm_embeddings.png", 
                            grid_size=(20, 20), 
                            image_size=(32, 32), 
                            upper_2d_points=None, 
                            lower_2d_points=None, 
                            l_arm_2d_points=None, 
                            r_arm_2d_points=None, 
                            zoom_factor=2):
    """
    Create a large image with small images representing the embedding points arranged in a grid.
    Optionally include red dots representing the bounding rectangles of 2D keypoints.
    :param skeletons: All normalized skeletons facing to the camera from bedlam dataset
    :param upper_embeddings: 2d embeddings from upper model
    :param lower_embeddings: 2d embeddings from lower model
    :param l_arm_embeddings: 2d embeddings from l_arm model
    :param r_arm_embeddings: 2d embeddings from r_arm model
    :param upper_body_indices: upper body indices representing target keypoints
    :param lower_body_indices: lower body indices representing target keypoints
    :param l_arm_body_indices: l_arm body indices representing target keypoints
    :param r_arm_body_indices: r_arm body indices representing target keypoints
    :param upper_filename: Filename for the upper body embeddings image.
    :param lower_filename: Filename for the lower body embeddings image.
    :param l_arm_filename: Filename for the left arm embeddings image.
    :param r_arm_filename: Filename for the right arm embeddings image.
    :param grid_size: The size of the grid (rows, columns).
    :param image_size: The size of each individual image in the grid.
    :param upper_2d_points: Optional list of 2D points for the upper body to include red dots.
    :param lower_2d_points: Optional list of 2D points for the lower body to include red dots.
    :param l_arm_2d_points: Optional list of 2D points for the left arm to include red dots.
    :param r_arm_2d_points: Optional list of 2D points for the right arm to include red dots.
    :param zoom_factor: Factor by which to zoom the images.
    """

    def draw_keypoints(draw, keypoints, indices, color='yellow', other_color='gray'):
        """Draw keypoints and bones on the image."""
        for i, j in BONES_COCO:
            if i in indices and j in indices:
                draw.line([keypoints[i], keypoints[j]], fill=color, width=10)
            else:
                draw.line([keypoints[i], keypoints[j]], fill=other_color, width=10)

    def create_grid_image(embedding, filename, points_2d=None, keypoint_indices=None):
        """Create a grid image for embedding points."""

        # Determine grid bounds
        x_min, y_min = np.min(embedding, axis=0)
        x_max, y_max = np.max(embedding, axis=0)

        # Create mesh grid
        x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
        y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)

        # Create an empty image for the grid
        grid_image = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]), (0, 0, 0))
        draw = ImageDraw.Draw(grid_image)

        # Assign images to grid cells
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x_start, x_end = x_bins[i], x_bins[i + 1]
                y_start, y_end = y_bins[j], y_bins[j + 1]

                # Find points within the current grid cell
                points_in_cell = [k for k in range(len(embedding)) if x_start <= embedding[k, 0] < x_end and y_start <= embedding[k, 1] < y_end]

                if points_in_cell:
                    # Choose a random point within the cell
                    chosen_index = random.choice(points_in_cell)
                    
                    # Create a black background image
                    img = Image.new('RGB', (640, 640), (0, 0, 0))

                    # Draw keypoints on the image
                    skeleton = skeletons[chosen_index]
                    keypoints = [(skeleton[idx][0], skeleton[idx][1]) for idx in range(len(skeleton))]
                    cell_draw = ImageDraw.Draw(img)
                    draw_keypoints(cell_draw, keypoints, keypoint_indices)

                    # Calculate zoomed size
                    zoomed_size = (int(image_size[0] * zoom_factor), int(image_size[1] * zoom_factor))
                    zoomed_img = img.resize(zoomed_size)

                    # Calculate crop box to center the zoomed image
                    left = (zoomed_size[0] - image_size[0]) / 2
                    top = (zoomed_size[1] - image_size[1]) / 2
                    right = (zoomed_size[0] + image_size[0]) / 2
                    bottom = (zoomed_size[1] + image_size[1]) / 2
                    crop_box = (left, top, right, bottom)

                    # Crop the zoomed image
                    cell_img = zoomed_img.crop(crop_box)

                    # Paste the cell image into the grid
                    grid_image.paste(cell_img, (i * image_size[0], j * image_size[1]))
                else:
                    # Fill the cell with black if no points are in the cell
                    img = Image.new('RGB', image_size, (0, 0, 0))
                    grid_image.paste(img, (i * image_size[0], j * image_size[1]))

        # Draw red dots for 2D points if provided
        if points_2d is not None:
            # Calculate the position in the grid image
            grid_x = int((points_2d[0] - x_min) / (x_max - x_min) * grid_size[0] * image_size[0])
            grid_y = int((points_2d[1] - y_min) / (y_max - y_min) * grid_size[1] * image_size[1])
            # Draw a red dot
            draw.ellipse((grid_x - 5, grid_y - 5, grid_x + 5, grid_y + 5), fill='red')

        # Save the resulting image
        grid_image.save(filename)
        print(f"Image {filename} saved")

        return [x_min, y_min, x_max, y_max]

    # Generate images for upper, lower, left arm, and right arm embeddings
    upper_body_range = create_grid_image(upper_embeddings, upper_filename, points_2d=upper_2d_points, keypoint_indices=upper_body_indices)
    lower_body_range = create_grid_image(lower_embeddings, lower_filename, points_2d=lower_2d_points, keypoint_indices=lower_body_indices)
    l_arm_range = create_grid_image(l_arm_embeddings, l_arm_filename, points_2d=l_arm_2d_points, keypoint_indices=l_arm_body_indices)
    r_arm_range = create_grid_image(r_arm_embeddings, r_arm_filename, points_2d=r_arm_2d_points, keypoint_indices=r_arm_body_indices)

def generate_embedding_images_kmeans(
                            centroids,
                            upper_centroids_embeddings,
                            lower_centroids_embeddings,
                            l_arm_centroids_embeddings,
                            r_arm_centroids_embeddings,
                            upper_body_indices,
                            lower_body_indices,
                            l_arm_body_indices,
                            r_arm_body_indices,
                            upper_filename="upper_embeddings.png", 
                            lower_filename="lower_embeddings.png", 
                            l_arm_filename="l_arm_embeddings.png", 
                            r_arm_filename="r_arm_embeddings.png", 
                            grid_size=(20, 20), 
                            image_size=(32, 32)):
    """
    Create a large image with images of centroid skeletons arranged in a grid based on their embeddings.
    Skeletons are scaled proportionally to fit within each cell and oriented correctly.
    """

    def draw_keypoints(draw, keypoints, highlight_indices, color='yellow', other_color='gray'):
        """Draw keypoints and bones on the image."""
        for i, j in BONES_COCO:
            if i in highlight_indices and j in highlight_indices:
                draw.line([(keypoints[i][0], keypoints[i][1]), (keypoints[j][0], keypoints[j][1])], fill=color, width=3)
            else:
                draw.line([(keypoints[i][0], keypoints[i][1]), (keypoints[j][0], keypoints[j][1])], fill=other_color, width=3)

    def scale_skeleton(keypoints_2d, image_size):
        """Scale the skeleton to fit within the image size while maintaining aspect ratio and correct orientation."""
        min_coords = np.min(keypoints_2d, axis=0)
        max_coords = np.max(keypoints_2d, axis=0)
        
        # Calculate the range of the skeleton in both dimensions
        skeleton_range = max_coords - min_coords
        
        # Calculate the scaling factor to fit the skeleton within the image
        scale = min(image_size[0] / skeleton_range[0], image_size[1] / skeleton_range[1])
        
        # Scale the skeleton
        scaled_keypoints = (keypoints_2d - min_coords) * scale
        
        # Flip the y-coordinates to correct the orientation
        scaled_keypoints[:, 1] = image_size[1] - scaled_keypoints[:, 1]
        
        # Calculate the offset to center the skeleton in the image
        offset_x = (image_size[0] - scaled_keypoints[:, 0].max()) / 2
        offset_y = (image_size[1] - scaled_keypoints[:, 1].max()) / 2
        
        scaled_keypoints[:, 0] += offset_x
        scaled_keypoints[:, 1] += offset_y
        
        return scaled_keypoints

    def create_grid_image(embeddings_centroid, filename, highlight_indices):
        """Create a grid image for embedding points."""

        # Determine grid bounds
        x_min, y_min = np.min(embeddings_centroid, axis=0)
        x_max, y_max = np.max(embeddings_centroid, axis=0)

        # Create mesh grid
        x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
        y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)

        # Create an empty image for the grid
        grid_image = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]), (0, 0, 0))

        # Assign images to grid cells
        for idx, embedding in enumerate(embeddings_centroid):
            # Determine which cell this embedding belongs to
            i = np.digitize(embedding[0], x_bins) - 1
            j = np.digitize(embedding[1], y_bins) - 1
            
            # Create a black background image
            img = Image.new('RGB', image_size, (0, 0, 0))

            # Draw keypoints on the image
            skeleton = centroids[idx]
            
            # Extract x and y coordinates, ignoring z
            keypoints_2d = skeleton[:, :2]
            
            # Scale the skeleton to fit within the image size
            scaled_keypoints = scale_skeleton(keypoints_2d, image_size)
            
            # Convert to integer coordinates
            keypoints = [(int(kp[0]), int(kp[1])) for kp in scaled_keypoints]
            
            cell_draw = ImageDraw.Draw(img)
            draw_keypoints(cell_draw, keypoints, highlight_indices)

            # Paste the cell image into the grid
            grid_image.paste(img, (i * image_size[0], j * image_size[1]))

        # Save the resulting image
        grid_image.save(filename)
        print(f"Image {filename} saved")

        return [x_min, y_min, x_max, y_max]

    # Generate images for upper, lower, left arm, and right arm embeddings
    upper_body_range = create_grid_image(upper_centroids_embeddings, upper_filename, upper_body_indices)
    lower_body_range = create_grid_image(lower_centroids_embeddings, lower_filename, lower_body_indices)
    l_arm_range = create_grid_image(l_arm_centroids_embeddings, l_arm_filename, l_arm_body_indices)
    r_arm_range = create_grid_image(r_arm_centroids_embeddings, r_arm_filename, r_arm_body_indices)

    return upper_body_range, lower_body_range, l_arm_range, r_arm_range