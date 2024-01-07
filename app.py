
import numpy as np 
import streamlit as st 
from PIL import Image
import os 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.saving import load_model

st.title("Malaria Detection")

model_path = os.path.join(os.getcwd(), 'cnn_model.h5')
cnn_model = load_model(model_path)

# Define a dictionary with image options and their corresponding file paths
image_options = {
    'Image 1': 'C33P1thinF_IMG_20150619_114756a_cell_179.png',
    'Image 2': 'C39P4thinF_original_IMG_20150622_105102_cell_83.png',
    'Image 3': 'C1_thinF_IMG_20150604_104722_cell_9.png',
    # Add more images as needed
}

# # Use a state variable to store the selected image
# selected_image = st.session_state.get("selected_image", None)

# # Create a grid layout with radio buttons for each image
# columns = st.columns(len(image_options))

# # Loop through each image option
# for i, (image_name, image_path) in enumerate(image_options.items()):
#     image = Image.open(image_path).resize((100, 100))
#     col = columns[i]
#     col.image(image, caption=image_name, use_column_width=False, width=200)

# # Use the selected_radio function to get the selected radio button value
#     if col.radio(f"Select", [image_name], key=f"radio_{image_name}"):
#         selected_image = image_name
#         st.session_state["selected_image"] = selected_image


    ####2 option

# # Display images and radio buttons in a horizontal layout
# columns = st.columns(len(image_options))
# for i, (image_name, image_path) in enumerate(image_options.items()):
#     image = Image.open(image_path).resize((100, 100))
#     col = columns[i]
#     col.image(image, caption=image_name, use_column_width=False, width=200)
    
#     # Add a unique key to each radio button
#     key = f"radio_{i}"
#     col.radio(f"Select {image_name}", [image_name], key=key)



    ###first option

 #Display images horizontally
row_images = [Image.open(image_path).resize((100, 100)) for image_path in image_options.values()]
st.image(row_images, caption=list(image_options.keys()), width=200)

# Create a radio button to choose the image
selected_image = st.radio("Choose an image:", list(image_options.keys()))


# Load the selected image
img_path = image_options[selected_image]
img = Image.open(img_path)

# Display the selected image
st.subheader("Selected Image:")
st.image(img, caption=f"{selected_image}", use_column_width=False, width=200)
st.write("")


# Allow users to upload their own image
uploaded_image = st.file_uploader("Upload your own image for checking", type=('jpg', 'jpeg', 'png','tiff'))

if uploaded_image is not None:
    st.subheader("Uploaded Image:")
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=False, width=200)
    st.write("")
    img=uploaded_image
    img=Image.open(img)

def cnn_make_prediction(img, cnn_model):
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res =(cnn_model.predict(input_img) > 0.5).astype("int32")
    if res:
        return "Parasitized"
    else:
        return "Uninfected"

if st.button("Detect"):
    result = cnn_make_prediction(img, cnn_model)
    st.subheader("Malaria Detection result")
    st.write(f"**{result}**")
