# Libraries
import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import openai
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch
import os
from dotenv import load_dotenv

# Object creation model, tokenizer, and processor from HuggingFace
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load environment variables
load_dotenv()
# Getting the key from env
openai.api_key = os.environ.get('OPENAI_API_KEY')  # Your OpenAI key
openai_model = "gpt-3.5-turbo"  # Updated to GPT-3.5-turbo for better performance

# Function to generate captions
def caption_generator(description):
    caption_prompt = f'''
    Please generate three unique and creative captions for Instagram for a photo that shows {description}.
    The captions should be fun and creative.
    Captions:
    1.
    2.
    3.
    '''
    
    response = openai.Completion.create(
        engine=openai_model,
        prompt=caption_prompt,
        max_tokens=200,
        temperature=0.7,
    )
    
    captions = response.choices[0].text.strip().split("\n")
    return captions

# Function to generate hashtags
def hashtag_generator(description):
    hashtag_prompt = f'''
    Please generate ten relevant and accurate hashtags for a photo that shows {description}. 
    The hashtags should be fun and creative.
    Format:
    #Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5 #Hashtag6 #Hashtag7 #Hashtag8 #Hashtag9 #Hashtag10
    '''
    
    response = openai.Completion.create(
        engine=openai_model,
        prompt=hashtag_prompt,
        max_tokens=100,
        temperature=0.7,
    )
    
    hashtags = response.choices[0].text.strip().split(" ")
    return hashtags

# Function to generate predictions from images
def prediction(img_list):
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    img = []
    
    for image in tqdm(img_list):
        i_image = Image.open(image)
        st.image(i_image, width=200)

        if i_image.mode != "RGB":
            i_image = i_image.convert("RGB")

        img.append(i_image)

    # Process the image to pixel values
    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    # Generate the caption from the model
    output = model.generate(pixel_val, **gen_kwargs)
    prediction = tokenizer.batch_decode(output, skip_special_tokens=True)
    prediction = [pred.strip() for pred in prediction]

    return prediction

# Display sample images and generate captions/hashtags
def sample():
    sample_images = {
        'Sample 1': 'image/beach.png',
        'Sample 2': 'image/coffee.png',
        'Sample 3': 'image/footballer.png',
        'Sample 4': 'image/mountain.jpg'
    }
    
    columns = cycle(st.columns(4))

    for img in sample_images.values():
        next(columns).image(img, width=150)

    for i, img in enumerate(sample_images.values()):
        if next(columns).button("Generate", key=i):
            description = prediction([img])
            st.subheader("Description for the Image:")
            st.write(description[0])

            st.subheader("Captions for this image:")
            captions = caption_generator(description[0])
            for caption in captions:
                st.write(caption)

            st.subheader("Hashtags:")
            hashtags = hashtag_generator(description[0])
            st.write(" ".join(hashtags))

# Function to handle image upload and generation
def upload():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        submit = st.form_submit_button("Generate")
        
        if submit and images:
            description = prediction(images)

            st.subheader("Description for the Image:")
            for i, caption in enumerate(description):
                st.write(caption)
                
            st.subheader("Captions for this image:")
            captions = caption_generator(description[0])
            for caption in captions:
                st.write(caption)
                
            st.subheader("Hashtags:")
            hashtags = hashtag_generator(description[0])
            st.write(" ".join(hashtags))

# Main function to run the app
def main():
    st.set_page_config(page_title="Caption and Hashtag Generator", layout="wide")
    st.title("Get Captions and Hashtags for your Image")

    tab1, tab2 = st.tabs(["Upload Image", "Sample"])

    with tab1:
        upload()

    with tab2:
        sample()

    st.subheader('By Varshith Kumar')

if __name__ == '__main__':
    main()
