import streamlit as st
import google.generativeai as genai
import io
from PIL import Image
import requests
from huggingface_hub import InferenceClient


# Function to navigate to another page
def navigate_to(page):
    st.session_state['page'] = page

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Define the Home Page
if st.session_state['page'] == 'home':

    # Create a styled box with a button inside
    st.markdown(
"""
<html>
<head>
<style>
.bounce-text {
  font-size: 3em;
  font-weight: bold;
  color: #ff5733;
  animation: bounce 2s infinite;
  text-align: center;
}
</style>
</head>
<body>
<div class="bounce-text">Explore Ai tools</div>
<div class="bounce-text">Brain ai</div>
<p></p>
</body>
</html>
""",unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

# Adding space between boxes by using margin in CSS
    box_style = """
<div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px border: 2px solid #4CAF50; 
                        border: 2px solid #4CAF50; border-radius: 20px ; aligin : center; margin-bottom: 20px; margin-right: 20px; ">
    <h3 style="color: #FFFFFF;"">{title}</h3>
    <p style="color: #FFFFFF;">{content}</p>

</div>
"""

    with col1:
        st.markdown(box_style.format(title="Brain GPT", content="Get answers. Find inspiration. Be more productive."), unsafe_allow_html=True)
        if st.button("GPT - 4 Model "):
            navigate_to('gemini_page')
    with col2:
        st.markdown(box_style.format(title="Image Generator", content="create realistic images and art from a description in natural language."), unsafe_allow_html=True)
        if st.button("Get Image"):
            navigate_to('image')
    with col3:
        st.markdown(box_style.format(title="Text Summarization", content="The passage is being summarized."), unsafe_allow_html=True)
        if st.button('Summarize It'):
            navigate_to('txt')

    with col4:
        st.markdown(box_style.format(title="Text Generation", content="Generate text by Microsoft"), unsafe_allow_html=True)
        if st.button('Generate Text'):
            navigate_to('Go')


# Define the Gemini Page
elif st.session_state['page'] == 'gemini_page':

    if st.button("Go Back to Home"):
        navigate_to('home')
    st.title("Welcome to GPT 4")
    # Configure Google Generative AI
    genai.configure(api_key="AIzaSyAWgwhw-ilTKqImvLwqTfcvZ2DgliUMUUk")

    # Input field for user's question
    text = st.text_input("Enter your question")

    # Instantiate the model
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])

    # Add a button to generate a response
    if st.button('Generate'):
        if text:
            response = chat.send_message(text)
            st.write(response.text)
        else:
            st.write("Please enter a question to generate a response.")

# Define the Image Generation Page
elif st.session_state['page'] == 'image':
    if st.button("Go Back to Home"):
        navigate_to('home')
        
    st.title("Image Generation")

    # Input field for prompt
    prompt = st.text_input('Enter prompt')

    # Define API details
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    # Function to query the API
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    if st.button('Generate Image'):
        if prompt:
            image_bytes = query({"inputs": prompt})
            image = Image.open(io.BytesIO(image_bytes))

            # Resize the image (for example, to 800x600 pixels)
            new_width = 300
            new_height = int((new_width / image.width) * image.height)
            image = image.resize((new_width, new_height))

            # Display the image
            st.image(image)

            # Add a download button
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            st.download_button(
                label="Download Image",
                data=buffer,
                file_name="generated_image.png",
                mime="image/png"
            )
        else:
            st.write("Please enter a prompt to generate an image.")


elif st.session_state['page'] == 'txt':
    if st.button("Go Back to Home"):
        navigate_to('home')
    st.title("Text Summarization")

    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    # Input field for text
    text_input = st.text_input("Enter the passage here:")
    
    if st.button('Generate'):
        if text_input:
            output = query({"inputs": text_input})
            st.write(output[0]['summary_text'])
        else:
            st.write("Please enter some text to summarize.")

elif st.session_state['page'] == 'Go':
    st.title("Microsoft ai text generation")

    if st.button("Go Back to Home"):
        navigate_to('home')

    # Add a button to generate a response
    client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="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",
)

    user_input = st.text_input("Enter your prompt")

    if st.button('Go'):
        generated_text = ""

        for message in client.chat_completion(
            messages=[{"role": "user", "content": user_input}],
            max_tokens=500,
            stream=True,
        ):
            generated_text += message.choices[0].delta.content

        st.write(generated_text)
