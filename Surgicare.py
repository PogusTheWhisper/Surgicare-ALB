import os
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Prevent torch.classes crash
import torch
import types
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

# Set environment variables
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

from utils.extract_wound_class import CachedWoundClassifier
from utils.extract_wound_features import CLIPWoundFeatureExtractor


@st.cache_resource
def load_model_components():
    classifier = CachedWoundClassifier()
    extractor = CLIPWoundFeatureExtractor()
    return classifier, extractor

classifier, extractor = load_model_components()


def analyze_wound(image_path, lang='th'):
    wound_class, probabilities = classifier.predict(image_path)
    features = extractor.extract_features(image_path, wound_class, lang=lang)
    result_lines = [f"Wound class: {wound_class}", "Top Features:"]
    result_lines += [f"{desc}: {score:.4f}" for desc, score in features]
    return "\n".join(result_lines)


def get_prompt(lang):
    if lang == "th":
        return """
คุณคือ "SurgiCare AI" และพูดแทนตัวเองว่าผม
ที่มีหน้าที่ "แนะนำวิธีรักษาแผลของคนไข้โดยวิเคราะจากข้อมูลที่คนไข้จะให้"

โดยพูดทวนลักษณะแผลของคนไข้เบื้องต้น 4-5 ข้อไม่ต้องลงรายละเอียด และแนะนำวิธีการดูแลแผลเป็นข้อๆ

** และคุณต้องเตือนเสมอว่าเป็นการแนะนำเบื้องต้น และถ้ามีอาการรุนแรงควรได้รับการตรวจสอบและยืนยันจากแพทย์ที่ดูแลโดยตรง **
"""
    else:
        return """
You are "SurgiCare AI", an AI assistant specialized in post-surgical wound care.
Your role is to give initial advice to patients based on the wound analysis they provide.

Summarize 4–5 key points of their wound condition briefly, then give step-by-step wound care instructions.

** Always remind the user that this is an initial suggestion and they should consult a licensed medical professional for serious symptoms. **
"""


def call_llm(api_key, model, max_tokens, temperature, top_p, history, user_input, lang):
    client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
    prompt = get_prompt(lang)

    messages = [{"role": "system", "content": prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
    except Exception as e:
        return f"<API_KEY_ERROR>: {str(e)}"

    return "".join(
        chunk.choices[0].delta.content
        for chunk in stream
        if hasattr(chunk, 'choices') and chunk.choices[0].delta.content
    )


def list_sample_images(directory):
    image_extensions = [".jpg", ".jpeg", ".png"]
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                rel_path = os.path.relpath(os.path.join(root, file), start=directory)
                all_images.append(rel_path)
    return sorted(all_images)


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def main():
    TYPHOON_API_KEY = st.secrets["TYPHOON_API_KEY"]

    session_defaults = {
        'llm_model': 'typhoon-v2-70b-instruct',
        'max_token': 512,
        'temperature': 0.6,
        'top_p': 0.95,
        'lang': 'th',
        'chat_history': [],
        'use_sample_image': False,
        'upload_image': None,
        'sample_image': None,
        'camera_image': None
    }

    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.sidebar:
        st.image('app_logo.png', width=140)
        st.title("Settings")
        st.session_state['lang'] = st.selectbox("Language", ["th", "en"], index=0)
        st.session_state['llm_model'] = st.selectbox("LLM Model", ["typhoon-v2-70b-instruct", "typhoon-v2.1-12b-instruct"])
        st.session_state['max_token'] = st.slider("Max Tokens", 50, 1024, st.session_state['max_token'], step=10)
        st.session_state['temperature'] = st.slider("Temperature", 0.0, 1.0, st.session_state['temperature'], step=0.05)
        st.session_state['top_p'] = st.slider("Top P", 0.0, 1.0, st.session_state['top_p'], step=0.05)

        if st.button("Clear Chat"):
            st.session_state['chat_history'] = []
            st.sidebar.success("Chat history cleared.")

    st.title("SurgiCare - AI Wound Assistant")
    st.markdown("Upload, select, or capture a wound image to receive analysis and care suggestions.")
    st.subheader("Image Input")

    camera_img = st.camera_input("Capture a wound image")
    if camera_img:
        st.session_state["camera_image"] = camera_img
        st.session_state["upload_image"] = None
        st.session_state["sample_image"] = None
        st.session_state["use_sample_image"] = False

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.session_state["upload_image"] = uploaded_img
        st.session_state["camera_image"] = None
        st.session_state["sample_image"] = None
        st.session_state["use_sample_image"] = False

    sample_dir = "careful_this_contain_wound_image"
    sample_images = list_sample_images(sample_dir)
    sample_labels = [f"Sample {i+1}" for i in range(len(sample_images))]
    sample_map = dict(zip(sample_labels, sample_images))

    if sample_images:
        selected_label = st.selectbox("Select a sample image", ["-- Select --"] + sample_labels)
        if selected_label != "-- Select --":
            selected_sample = sample_map[selected_label]
            st.session_state["sample_image"] = os.path.join(sample_dir, selected_sample)
            st.session_state["use_sample_image"] = True
            st.session_state["camera_image"] = None
            st.session_state["upload_image"] = None

    image = None
    if st.session_state["use_sample_image"] and st.session_state["sample_image"]:
        image = st.session_state["sample_image"]
    elif st.session_state["upload_image"]:
        image = st.session_state["upload_image"]
    elif st.session_state["camera_image"]:
        image = st.session_state["camera_image"]

    if image:
        img = Image.open(image)
        st.markdown(
            f"""
            <div style="text-align:center">
                <img src="data:image/jpeg;base64,{image_to_base64(img)}" width="300">
                <p><strong>Selected Image</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Analyze Wound"):
        if not image:
            st.warning("Please provide a wound image.")
        else:
            analysis = analyze_wound(image, lang=st.session_state['lang'])
            st.text_area("Wound Analysis", analysis, height=200)

            # Add analysis to chat history as starting point
            st.session_state['chat_history'] = [{"role": "assistant", "content": analysis}]

            response = call_llm(
                TYPHOON_API_KEY,
                st.session_state['llm_model'],
                st.session_state['max_token'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                st.session_state['chat_history'],
                "โปรดให้คำแนะนำเพิ่มเติมในการดูแลแผลตามลักษณะด้านบน",
                lang=st.session_state['lang']
            )

            # with st.chat_message("assistant"):
            #     st.write(response)

            st.session_state['chat_history'].append({"role": "assistant", "content": response})

    # ==== Chat Input ====
    prompt = st.chat_input("Ask more about your wound..." if st.session_state['lang'] == "en" else "พิมพ์คำถามเพิ่มเติมเกี่ยวกับแผลของคุณ...")

    # Initialize chat history if not already
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # ==== Display Chat History ====
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # ==== Handle New Prompt ====
    if prompt:
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        # Get assistant reply
        reply = call_llm(
            TYPHOON_API_KEY,
            st.session_state['llm_model'],
            st.session_state['max_token'],
            st.session_state['temperature'],
            st.session_state['top_p'],
            st.session_state['chat_history'],
            prompt,
            lang=st.session_state['lang']
        )

        # Show assistant message
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state['chat_history'].append({"role": "assistant", "content": reply})

    # ==== Download Chat Log Button ====
    if st.session_state['chat_history']:
        chat_log = "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state['chat_history'])
        st.download_button(
            label="Download Session Log",
            data=chat_log,
            file_name="surgicare_session_log.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
