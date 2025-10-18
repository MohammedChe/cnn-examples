import os
# Force CPU-only TensorFlow and reduce log noise
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # strongly disable CUDA visibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # 0=INFO,1=WARNING,2=ERROR,3=FATAL (hide ERROR logs)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # optional: disable oneDNN optimizations



import numpy as np
import sys
import time
import argparse

# We'll import tensorflow when needed and import pygame only for GUI mode.


try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# Parse command-line arguments
parser = argparse.ArgumentParser(description="MNIST recognition demo (GUI and headless test)")
parser.add_argument("model", help="Path to the Keras .h5 model file")
parser.add_argument("--test-load", action="store_true", help="Headless: check model file and try loading it without opening pygame")
args = parser.parse_args()

# If user requested a headless test-load, try to load the model (if tensorflow is available)
if args.test_load:
    model_path = args.model
    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}")
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow is not available in this environment. To fully test model loading, install TensorFlow or run the GUI where available.")
        print(f"Model file exists: {model_path}")
        sys.exit(0)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully: {model_path}")
        sys.exit(0)
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

# Otherwise proceed with GUI mode. Load TensorFlow model now.
import tensorflow as tf
model = tf.keras.models.load_model(args.model, compile=False)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame (GUI mode)
try:
    import pygame
except ModuleNotFoundError:
    sys.exit("pygame is not installed in this Python environment. Install it (e.g. `pip install pygame`) or run with --test-load to test model loading without GUI.")

pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "./apps/assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

while True:

    # Check if game quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(handwriting).reshape(1, 28, 28, 1)]
        ).argmax()

    # Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
