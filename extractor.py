import cv2
import numpy as np
import os

# Define paths
image_folder = "./resources/png/"
mask_path = "./resources/mask_square_2.png"
# Does rotation correction needed?
rotate = False
# Load the mask in grayscale
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Function to rotate the image by 180 degrees
def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_180)

# Function to display instructions on the image
def display_instructions(image, info_text):
    image_copy = image.copy()
    cv2.putText(image_copy, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("Image Viewer", image_copy)

# Function to process the image
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image {image_path}")
        return

    # Step 1: Display image and wait for key press
    while rotate:
        instructions = (
            "Press 'r' to rotate 180 degrees and save\n"
            "Press 's' to skip this image\n"
            "Press 'q' to quit\n"
            "Press 'Enter' or 'Space' to move to next step"
        )
        display_instructions(image, instructions)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):  # Rotate image by 180 degrees and save
            print("Rotating image by 180 degrees...")
            image = rotate_image(image)
            cv2.imwrite(image_path, image)
            cv2.destroyAllWindows()
            break  # Close and move to the next image
        elif key == ord('s'):  # Skip image
            print("Skipping image...")
            return  # Skip and move to the next image
        elif key == ord('q'):  # Quit the program
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()
        elif key in [13, 32]:  # Enter or Space to proceed
            break

    # Step 2: Select two corners
    selected_points = []

    def select_points(event, x, y, flags, param):
        nonlocal selected_points

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) == 2:
                selected_points[0] = selected_points[1]
                selected_points[1] = (x, y)
            else:
                selected_points.append((x, y))

            # Reload the image each time a point is selected to clear old rectangles
            image_copy = cv2.imread(image_path)

            # Draw red square (16x16) at the selected point
            for point in selected_points:
                cv2.rectangle(image_copy, (point[0] - 8, point[1] - 8), (point[0] + 8, point[1] + 8), (0, 0, 255), -1)

            display_instructions(image_copy, instructions)

    # Step 2.1: Display image for point selection
    instructions = (
        "Select two points (top-left and bottom-right)\n"
        "Press 'q' to quit\n"
        "Press 'n' to restart selection\n"
        "Press 'Enter' or 'Space' to continue"
    )
    display_instructions(image, instructions)
    cv2.setMouseCallback("Image Viewer", select_points)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()
        elif key == ord('n'):  # Restart the selection
            selected_points = []
            image = cv2.imread(image_path)  # Reload the original image
            display_instructions(image, instructions)
        elif key in [13, 32]:  # Enter or Space to continue
            break

    # Step 3: Compute transformation matrix and apply
    top_left = selected_points[0]
    bottom_right = selected_points[1]
    mask_height, mask_width = mask.shape

    mask_points = np.array([[0, 0], [mask_width - 1, 0], [mask_width - 1, mask_height - 1], [0, mask_height - 1]], dtype='float32')
    selected_points_array = np.array([top_left, [bottom_right[0], top_left[1]], bottom_right, [top_left[0], bottom_right[1]]], dtype='float32')

    # Compute the perspective transform
    M = cv2.getPerspectiveTransform(selected_points_array, mask_points)

    # Apply the perspective transformation to the image
    warped_image = cv2.warpPerspective(image, M, (mask_width, mask_height+270*2))

    # Step 4: Binarize the mask and find rectangles
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
    rectangles.sort(key=lambda x: (x[1], x[0]))  # Sort rectangles top to bottom, left to right

    # Step 5: Extract and save rectangles
    output_folder = "output_rect"
    output_folder_sq = "output_rect_square"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_sq, exist_ok=True)

    padding = 4
    for idx, (x, y, w, h) in enumerate(rectangles):
        img_name = os.path.basename(image_path).split('.')[0]

        cropped_rect = warped_image[y:y + h + 270, x:x + w]
        output_filename = f"{output_folder}/{img_name}_{idx + 1}.png"
        cv2.imwrite(output_filename, cropped_rect)

        cropped_rect_sq = warped_image[y:y + h, x:x + w]
        output_filename_sq = f"{output_folder_sq}/{img_name}_{idx + 1}.png"
        cv2.imwrite(output_filename_sq, cropped_rect_sq)

        print(f"Saved: {output_filename}")

    # Visualize selected rectangles
    image_with_rectangles = warped_image.copy()
    for x, y, w, h in rectangles:
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Selected Rectangles", image_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Loop through images in the folder and process each one
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {image_path}")
        process_image(image_path)