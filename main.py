import cv2
import os

# Function to apply high-pass filter
def apply_high_pass_filter(image):
    high_pass = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(high_pass)

# Function to apply low-pass filter
def apply_low_pass_filter(image):
    low_pass = cv2.GaussianBlur(image, (5, 5), 0)
    return low_pass

# Function to apply histogram equalization
def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Function to process images in the dataset
def process_images(dataset_folder):
    for class_folder in os.listdir(dataset_folder):
        class_folder_path = os.path.join(dataset_folder, class_folder)

        # Ensure the path is a directory
        if os.path.isdir(class_folder_path):
            # Create output folders if they don't exist for each class
            os.makedirs(os.path.join("output", "high_pass", class_folder), exist_ok=True)
            os.makedirs(os.path.join("output", "low_pass", class_folder), exist_ok=True)
            os.makedirs(os.path.join("output", "histogram_equalization", class_folder), exist_ok=True)

            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)

                # Read the image
                image = cv2.imread(file_path)

                # Check if the image is read correctly
                if image is None:
                    print("Error: Unable to read the image at", file_path)
                    continue  # Skip to the next iteration if there's an issue with the image

                # Apply high-pass filter
                high_pass_result = apply_high_pass_filter(image)

                # Apply low-pass filter
                low_pass_result = apply_low_pass_filter(image)

                # Apply histogram equalization
                equalized_result = apply_histogram_equalization(image)

                # Save the results sequentially
                cv2.imwrite(os.path.join("output", "high_pass", class_folder, filename), high_pass_result)
                cv2.imwrite(os.path.join("output", "low_pass", class_folder, filename), low_pass_result)
                cv2.imwrite(os.path.join("output", "histogram_equalization", class_folder, filename), equalized_result)

# Main function
if __name__ == "__main__":
    dataset_folder = 'data'

    # Process images in the dataset
    process_images(dataset_folder)
