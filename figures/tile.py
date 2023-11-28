from PIL import Image
import os, math

def tile_images(input_folder, output_path, tile_size=(100, 100), padding=10, columns=9):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    # Calculate the number of rows needed based on the number of images
    rows = math.ceil(len(image_files) / columns)

    # Open the first image to get its size
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = Image.open(first_image_path)
    image_width, image_height = first_image.size

    # Calculate the size of the output image
    output_width = columns * (tile_size[0] + padding) - padding
    output_height = rows * (tile_size[1] + padding) - padding

    # Create a new image with the calculated size
    output_image = Image.new('RGB', (output_width, output_height), (255, 255, 255))

    # Paste each image onto the output image
    current_x = 0
    current_y = 0
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        # Resize the image to the specified tile size
        image = image.resize(tile_size)

        # Paste the image onto the output image
        output_image.paste(image, (current_x, current_y))

        # Update the current_x and current_y positions for the next image
        current_x += tile_size[0] + padding
        if current_x >= output_width:
            current_x = 0
            current_y += tile_size[1] + padding

    # Save the final tiled image
    output_image.save(output_path)

if __name__ == "__main__":
    input_folder = "."
    output_path = "./tiled_image.png"

    tile_images(input_folder, output_path)
