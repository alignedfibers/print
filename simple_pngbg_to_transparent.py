import os
from rembg import remove
from PIL import Image

# Force CPU processing
os.environ["U2NET_ENABLE_GPU"] = "False"

input_path = "greentreesinbackground_2752732759.png"  # Replace with your file path
output_path = "output_greentreesinbackground_2752732759.png"

# Open the image
image = Image.open(input_path)

# Remove the background
output = remove(image)

# Save the processed image
output.save(output_path)

print(f"Background removed and saved as: {output_path}")
