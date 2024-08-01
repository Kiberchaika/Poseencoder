import cv2
import pygame
import numpy as np
from ultralytics import YOLO
from pose_2d_regressor_pca import PoseRegressionModel
import torch
import numpy

# Load the trained Pose 2D regressor model
regressor_model = PoseRegressionModel(encoders=None, input_dim=34, output_dim=8)
regressor_model.load_model('best__model.pth')
regressor_model.model.eval()

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Human Pose Detection with Tension Metrics")

# Load the YOLO model
model = YOLO("yolov8n-pose.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

running = True
clock = pygame.time.Clock()


# Define the connections between landmarks
CONNECTIONS = [
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

def draw_pca_circles(screen, embeddings):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # R, G, B, Y
    labels = ['upper', 'lower', 'l_arm', 'r_arm']
    circle_radius = 30  # Adjust this value to change the size of the circles

    for i in range(4):
        x = int(embeddings[i*2] * width)
        y = int(embeddings[i*2+1] * height)
        pygame.draw.circle(screen, colors[i], (x, y), circle_radius)
        
        # Optionally, add labels
        font = pygame.font.Font(None, 24)
        text = font.render(labels[i], True, (255, 255, 255))
        screen.blit(text, (x - 20, y + circle_radius + 5))

def process_landmarks(landmarks):
    # Center the landmarks
    center = landmarks.mean(axis=0)
    centered_landmarks = landmarks - center

    # Calculate the scale to fit Y exactly from -1 to 1
    y_min = np.min(centered_landmarks[:, 1])
    y_max = np.max(centered_landmarks[:, 1])
    y_scale = 2 / (y_max - y_min)

    # Scale both X and Y by the same factor to maintain aspect ratio
    normalized_points = centered_landmarks * y_scale

    # Shift Y to ensure it's exactly in the range [-1, 1]
    y_shift = -(np.min(normalized_points[:, 1]) + np.max(normalized_points[:, 1])) / 2
    normalized_points[:, 1] += y_shift

    normalized_points[:,1] = - normalized_points[:,1]

    # Flatten the normalized points to match the input shape expected by the model
    flattened_points = normalized_points.reshape(1, -1)

    # Convert to torch tensor
    input_tensor = torch.FloatTensor(flattened_points).to(regressor_model.device)

    # Make prediction
    with torch.no_grad():
        predicted_embeddings = regressor_model.model(input_tensor)

    # print(f"Y range: {normalized_points[:, 1].min():.2f} to {normalized_points[:, 1].max():.2f}")
    return predicted_embeddings.cpu().numpy()[0], normalized_points

    return predicted_embeddings.cpu().numpy()[0], normalized_points

def draw_normalized_landmarks(screen, normalized_landmarks):
    rect_size = 150
    rect_pos = (width - rect_size - 10, 10)  # 10 pixels padding from the top-right corner
    
    # Draw the rectangle
    pygame.draw.rect(screen, (200, 200, 200), (*rect_pos, rect_size, rect_size), 2)
    
    # Draw axes labels
    font = pygame.font.Font(None, 20)
    screen.blit(font.render('-1', True, (255, 255, 255)), (rect_pos[0] - 20, rect_pos[1] + rect_size // 2))
    screen.blit(font.render('1', True, (255, 255, 255)), (rect_pos[0] + rect_size + 5, rect_pos[1] + rect_size // 2))
    screen.blit(font.render('-1', True, (255, 255, 255)), (rect_pos[0] + rect_size // 2, rect_pos[1] + rect_size + 5))
    screen.blit(font.render('1', True, (255, 255, 255)), (rect_pos[0] + rect_size // 2, rect_pos[1] - 20))
    screen.blit(font.render('X', True, (255, 255, 255)), (rect_pos[0] + rect_size + 5, rect_pos[1] + rect_size + 5))
    screen.blit(font.render('Y', True, (255, 255, 255)), (rect_pos[0] - 20, rect_pos[1] - 20))

    # Draw landmarks
    if not numpy.isnan(normalized_landmarks).any():
        for x, y in normalized_landmarks:
            screen_x = int(rect_pos[0] + (x + 1) * rect_size / 2)
            screen_y = int(rect_pos[1] + (1 - y) * rect_size / 2)  # Flip Y-axis
            pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 2)

def resize_and_pad(image, target_size):
    ih, iw = image.shape[:2]
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))
    image_padded = np.full((h, w, 3), 0, dtype=np.uint8)
    dw, dh = (w - nw) // 2, (h - nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    return image_padded, scale, (dw, dh)

def get_color(value):
    # Map value (0-1) to a color gradient (blue to red)
    return (int(value * 255), 0, int((1 - value) * 255))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and pad the frame
    frame_resized, scale, (dx, dy) = resize_and_pad(frame, (width, height))

    # Run YOLO detection on the resized frame
    results = model(frame_resized, verbose=False)

    # Convert the frame to RGB (Pygame uses RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Create a Pygame surface from the frame
    py_image = pygame.image.frombuffer(frame_rgb.tostring(), frame_rgb.shape[1::-1], "RGB")

    # Draw the frame on the Pygame window
    screen.blit(py_image, (0, 0))

    for result in results:
        if result.keypoints is not None:
            landmarks = result.keypoints.xy[0].cpu().numpy()

            if len(landmarks) < 17:
                continue
            
            # Process landmarks and get predictions
            predicted_embeddings, normalized_landmarks = process_landmarks(landmarks)
            
            # Draw the PCA circles
            if not numpy.isnan(predicted_embeddings).any():
                draw_pca_circles(screen, predicted_embeddings)

            # Draw the normalized landmarks
            draw_normalized_landmarks(screen, normalized_landmarks)

            # Draw original landmarks
            for x, y in landmarks:
                pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 3)

            # # Draw the PCA circles
            # if not numpy.isnan(predicted_embeddings).any():
            #     draw_pca_circles(screen, predicted_embeddings)
            
            # Use predicted_embeddings as needed
            # For example, you can print them:
            # print("Predicted embeddings:", predicted_embeddings)

            # # Draw landmarks
            # for x, y in landmarks:
            #     pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 3)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)

# Clean up
cap.release()
pygame.quit()