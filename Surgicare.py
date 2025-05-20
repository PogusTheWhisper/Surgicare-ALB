import streamlit as st
import os
from openai import OpenAI
from utils.extract_wound_class import WoundClassificationModel
from utils.extract_wound_features import extract_wound_features
from utils.analyze_wound_features import analyze_wound_features

def analyze_wound(img_path, lang='th', model_name='SurgiCare-V1-large-turbo'):
    classifier = WoundClassificationModel()
    classifier.load_model(model_name)

    wound_class, predictions = classifier.extract_wound_class(img_path)
    wound_features = extract_wound_features(img_path)
    wound_assessment = analyze_wound_features(wound_features, lang)

    result_lines = [
        f"wound class: {wound_class}",
        *[f"{key}: {value}" for key, value in wound_assessment.items()]
    ]

    return "\n".join(result_lines)

def call_llm(api_key, model, max_tokens, temperature, top_p, user_input):
    client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    คุณคือ "SurgiCare AI" และพูดแทนตัวเองว่าผม
                    ที่มีหน้าที่ "แนะนำวิธีรักษาแผลของคนไข้โดยวิเคราะจากข้อมูลที่คนไข้จะให้"
                    
                    โดยพูดทวนลักษณะแผลของคนไข้เบื้องต้น 4-5 ข้อไม่ต้องลงรายละเอียด และแนะนำวิธีการดูแลแผลเป็นข้อๆ
                    
                    ** และคุณต้องเตือนเสมอว่าเป็นการแนะนำเบื้องต้น และถ้ามีอาการรุนแรงควรได้รับการตรวจสอบและยืนยันจากแพทย์ที่ดูแลโดยตรง **
                    """,
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
    except Exception as e:
        return f'<API_KEY_ERROR>: {str(e)}'

    return "".join(
        chunk.choices[0].delta.content
        for chunk in stream if hasattr(chunk, 'choices') and chunk.choices[0].delta.content
    )

def list_sample_images(directory):
    """List all image files in the given directory."""
    image_extensions = [".jpg", ".jpeg", ".png"]
    return sorted(f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in image_extensions)

def main():
    # Initialize configuration
    TYPHOON_API_KEY = st.secrets["TYPHOON_API_KEY"]   
             
    classifier = WoundClassificationModel()
    classifier.load_model('SurgiCare-V1-large-turbo')

    # Default session state values
    session_defaults = {
        'classify_model': 'SurgiCare-V1-large-turbo',
        'llm_model': 'typhoon-v2-70b-instruct',
        'max_token': 512,
        'temperature': 0.6,
        'top_p': 0.95,
        'full_diagnose_w_chat': [],
        'chat_history': [],
        'use_sample_image': None,
        'upload_image': None,
        'sample_image': None,
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar configuration
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.image('app_logo.png', width=125)
        
        st.title("Config")
        st.markdown('You can lower temperature to make it more deterministic.')

        classify_model = st.selectbox(
            "Classify Model",
            options=["SurgiCare-V1-large-turbo", "SurgiCare-V1-large", "SurgiCare-V1-mini-medium", "SurgiCare-V1-mini-small"],
            index=["SurgiCare-V1-large-turbo", "SurgiCare-V1-large", "SurgiCare-V1-mini-medium", "SurgiCare-V1-mini-small"].index(st.session_state['classify_model'])
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["typhoon-v2-70b-instruct", "typhoon-v2-8b-instruct"],
            index=["typhoon-v2-70b-instruct", "typhoon-v2-8b-instruct"].index(st.session_state['llm_model'])
        )
        
        max_token = st.slider('Max Token', 50, 512, st.session_state['max_token'], step=10)
        temperature = st.slider('Temperature', 0.0, 1.0, st.session_state['temperature'], step=0.05)
        top_p = st.slider('Top P', 0.0, 1.0, st.session_state['top_p'], step=0.05)

        if st.button('Save Config'):
            st.session_state.update({
                'classify_model': classify_model,
                'llm_model': llm_model,
                'max_token': max_token,
                'temperature': temperature,
                'top_p': top_p
            })
            st.success("Configuration saved!")
            classifier = WoundClassificationModel()
            classifier.load_model(classify_model)

    st.title("Welcome to SurgiCare!!")
    st.markdown("AI Application for supporting post-surgery patient recovery.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Take a picture")
        img_file_buffer = st.camera_input("Camera input")
        if img_file_buffer:
            user_content = analyze_wound(img_file_buffer, model_name=st.session_state['classify_model'])
            diagnose_response = call_llm(
                TYPHOON_API_KEY,
                st.session_state['llm_model'],
                st.session_state['max_token'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                user_content
            )
            st.session_state['full_diagnose_w_chat'].append({"role": "assistant", "content": diagnose_response})
            st.session_state['chat_history'].append({"role": "assistant", "content": diagnose_response})

    with col2:
        st.header("Upload or select a sample photo")
        
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.session_state['upload_image'] = uploaded_file

        sample_images_dir = "careful_this_contain_wound_image/"
        sample_images = list_sample_images(sample_images_dir)
        if sample_images:
            selected_sample = st.selectbox("Or use a sample image", options=["-- Select --"] + sample_images)
            st.session_state['sample_image'] = os.path.join(sample_images_dir, selected_sample) if selected_sample != "-- Select --" else None

        if st.toggle("Switch to use sample images"):
            st.session_state["use_sample_image"] = True

        if st.button("Process Image"):
            if st.session_state['use_sample_image'] and st.session_state['sample_image']:
                user_content = analyze_wound(st.session_state['sample_image'], model_name=st.session_state['classify_model'])
            elif not st.session_state['use_sample_image'] and st.session_state['upload_image']:
                user_content = analyze_wound(st.session_state['upload_image'], model_name=st.session_state['classify_model'])
            else:
                st.warning("Please select or upload an image first.")
                return

            diagnose_response = call_llm(
                TYPHOON_API_KEY,
                st.session_state['llm_model'],
                st.session_state['max_token'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                user_content
            )
            st.session_state['full_diagnose_w_chat'].append({"role": "assistant", "content": diagnose_response})
            st.session_state['chat_history'].append({"role": "assistant", "content": diagnose_response})
            
            # Clear processed images
            st.session_state['sample_image'] = None
            st.session_state['upload_image'] = None

    # Display chat history
    st.subheader("Full Diagnose History")
    for msg in st.session_state['full_diagnose_w_chat']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    prompt = st.chat_input("พิมพ์คำถามตรงนี้!!")
    if prompt:
        st.session_state['full_diagnose_w_chat'].append({"role": "user", "content": prompt})
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        conversation_history = "\n".join(msg["content"] for msg in st.session_state['chat_history'])

        response = call_llm(
            TYPHOON_API_KEY,
            st.session_state['llm_model'],
            st.session_state['max_token'],
            st.session_state['temperature'],
            st.session_state['top_p'],
            conversation_history
        )

        # Update histories with the assistant's response
        st.session_state['full_diagnose_w_chat'].append({"role": "assistant", "content": response})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # Clear chat display and show updated history
        st.session_state['chat_history'] = []
        st.session_state['full_diagnose_w_chat'] = []

if __name__ == "__main__":
    main()
