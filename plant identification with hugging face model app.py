import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import numpy as np
import pandas as pd

# Load pre-trained model and feature extractor from Hugging Face
model_name = "google/vit-base-patch16-224-in21k"  # Vision Transformer model pre-trained on ImageNet21k
model = AutoModelForImageClassification.from_pretrained(model_name)
extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Load plant info database (replace with your actual CSV)
plant_info_df = pd.read_csv('plant_info.csv')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert image to RGB (if it's in another format)
    image = image.convert("RGB")
    
    # Use the feature extractor to process the image (resize, normalize, etc.)
    inputs = extractor(images=image, return_tensors="pt")
    return inputs

# Function to predict plant species and get care information
def predict_plant(image):
    inputs = preprocess_image(image)
    
    # Make prediction using the Hugging Face model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class index (the model's output)
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    
    # Map class index to plant name (in real usage, you would need to use actual plant labels)
    plant_name = plant_info_df.iloc[predicted_class_idx]['plant_name']
    care_tips = plant_info_df.iloc[predicted_class_idx]['care_tips']
    water_level = plant_info_df.iloc[predicted_class_idx]['water_level']
    
    return plant_name, care_tips, water_level

# Streamlit UI components
st.title("Plant & Flower Identification App")

st.write(
    """
    Upload a picture of a plant or flower, and we will help you identify it, 
    provide care tips, and suggest the water level!
    """
)

# Image upload
uploaded_image = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image with PIL
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions when the user uploads an image
    if st.button('Identify Plant'):
        plant_name, care_tips, water_level = predict_plant(image)

        # Display results
        st.write(f"### Plant Species: {plant_name}")
        st.write(f"### Care Tips: {care_tips}")
        st.write(f"### Suggested Water Level: {water_level}")
else:
    st.write("Please upload an image to start.")




import pandas as pd

# Create a simple DataFrame with plant information
plant_info = {
    'plant_name': ['Rose', 'Tulip', 'Sunflower', 'Cactus', 'Orchid'],
    'care_tips': [
        'Keep soil moist but not soggy',
        'Water regularly, but not excessively',
        'Water generously during the growing season',
        'Water sparingly, very low',
        'Water once a week'
    ],
    'water_level': ['Moderate', 'Low', 'High', 'Very low', 'Low']
}

plant_info_df = pd.DataFrame(plant_info)

# Save it to CSV
plant_info_df.to_csv('plant_info.csv', index=False)
