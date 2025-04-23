# import cv2

# # Read the image from the current directory
# img = cv2.imread("soil_image.jpg")

# # Check if image is loaded properly
# if img is None:
#     print("❌ Failed to load image. Check the file name and path.")
# else:
#     # Resize to 800x600
#     resized = cv2.resize(img, (800, 600))

#     # Save the resized image
#     cv2.imwrite("soil_image_resized.jpg", resized)

#     print("✅ Resized image saved as soil_image_resized.jpg")



import cv2

# Read the image from the specified path
img = cv2.imread("myplant.jpg")

# Check if image is loaded properly
if img is None:
    print("❌ Failed to load image. Check the file name and path.")
else:
    # Resize to 800x600 (you can change the size if needed)
    resized = cv2.resize(img, (800, 600))

    # Save the resized image
    cv2.imwrite("myimage_resized.jpg", resized)

    print("✅ Image resized and saved as myplant_resized.jpg")



# import cv2

# # Read the image from the current directory
# img = cv2.imread("myplant_2.jpg")  

# # Check if image is loaded properly
# if img is None:
#     print("❌ Failed to load image. Check the file name and path.")
# else:
#     # Resize to 800x600
#     resized = cv2.resize(img, (800, 600))

#     # Save the resized image
#     cv2.imwrite("plant_detection/myplant_2_resized.jpg", resized)
#     print("✅ Image resized and saved as myplant_2_resized.jpg")


# import cv2

# # Read the image from the current directory
# img = cv2.imread("drysoil.jpg")  

# # Check if image is loaded properly
# if img is None:
#     print("❌ Failed to load image. Check the file name and path.")
# else:
#     # Resize to 800x600
#     resized = cv2.resize(img, (800, 600))

#     # Save the resized image
#     cv2.imwrite("drysoil_resized.jpg", resized)
#     print("✅ Image resized and saved as drysoil_resized.jpg")