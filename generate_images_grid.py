def generate_embedding_images(self, 
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
                    skeleton = self.skeletons[chosen_index]
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
    self.upper_body_range = create_grid_image(self.upper_embedding, upper_filename, points_2d=upper_2d_points, keypoint_indices=self.upper_body_indices)
    self.lower_body_range = create_grid_image(self.lower_embedding, lower_filename, points_2d=lower_2d_points, keypoint_indices=self.lower_body_indices)
    self.l_arm_range = create_grid_image(self.l_arm_embedding, l_arm_filename, points_2d=l_arm_2d_points, keypoint_indices=self.l_arm_body_indices)
    self.r_arm_range = create_grid_image(self.r_arm_embedding, r_arm_filename, points_2d=r_arm_2d_points, keypoint_indices=self.r_arm_body_indices)
