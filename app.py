
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="banana_leaf_multiclass.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (MUST match class_indices)
labels = [
    "Cordana",
    "Healthy",
    "Panama Disease",
    "Sigatoka"
]

st.set_page_config(page_title="Banana Leaf Disease Detector", layout="centered")

st.title("🍌 Banana Leaf Disease Detection")
st.write("Upload a banana leaf image to detect the disease.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    disease = labels[idx]
    confidence = preds[idx] * 100

    st.subheader("🧪 Diagnosis")
    st.success(f"**{disease}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# loading the tflite model
interp = tf.lite.Interpreter(model_path="banana_leaf_multiclass.tflite")
interp.allocate_tensors()

inp_details = interp.get_input_details()
out_details = interp.get_output_details()

class_labels = ["Cordana", "Healthy", "Panama Disease", "Sigatoka"]

st.set_page_config(
    page_title="Banana Leaf Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# spent way too long on the styling but it looks good now
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa, #e8f5e9);
        font-family: Poppins, sans-serif;
    }
    
    h1 {
        color: #2d6a4f;
        font-weight: bold;
        text-align: center;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-size: 2.5rem !important;
    }
    
    h3 {
        color: #2d6a4f;
        font-weight: 600;
    }
    
    .subtitle {
        text-align: center;
        color: #52b788;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        color: #2d6a4f;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(82, 183, 136, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
        color: white !important;
    }
    
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #1b4332;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #52b788;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* file upload styling */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border-radius: 20px;
        padding: 0;
    }
    
    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: linear-gradient(135deg, rgba(45, 106, 79, 0.15), rgba(82, 183, 136, 0.15)) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px dashed rgba(82, 183, 136, 0.6);
        padding: 3rem 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    
    [data-testid="stFileUploader"] section:hover {
        background: linear-gradient(135deg, rgba(45, 106, 79, 0.25), rgba(82, 183, 136, 0.25)) !important;
        border-color: #52b788;
        box-shadow: 0 12px 32px rgba(82, 183, 136, 0.3);
        transform: translateY(-2px);
    }
    
    [data-testid="stFileUploader"] section > div {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #2d6a4f !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #52b788 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 50%, #52b788 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 0.5rem;
    }
    
    img {
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    img:hover {
        transform: scale(1.02);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #52b788, #2d6a4f);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background-color: rgba(82, 183, 136, 0.2);
        border-radius: 10px;
    }
    
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 10px rgba(82, 183, 136, 0.3);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(82, 183, 136, 0.4);
    }
    
    .markdown-text-container {
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 15px;
        padding: 1rem;
    }
    
    .streamlit-expanderHeader {
        background: rgba(82, 183, 136, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stCaption {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# sidebar with model specs
with st.sidebar:
    st.markdown("### 🔬 Model Information")
    st.markdown("---")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Training:** Transfer Learning")
    st.write("**Inference:** Offline (TensorFlow Lite)")
    st.write("**Classes:** 4 banana leaf diseases")
    st.write("**Input Size:** 224×224 RGB")
    
    st.markdown("---")
    
    st.write("**📱 Designed for:**")
    st.write("Low-end smartphones with limited connectivity")
    
    st.write("**⚡ Inference Time:**")
    st.write("< 100ms on mobile devices")
    
    st.write("**🎯 Use Case:**")
    st.write("Field diagnosis for smallholder farmers")
    
    st.markdown("---")
    st.caption("🏆 Built for Hackathon Demo")

st.markdown("<h1>🌿 Banana Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Plant Health Monitoring for Farmers</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Upload a Banana Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", width=250)
    
    with col2:
        # preprocess
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # run inference
        interp.set_tensor(inp_details[0]['index'], img)
        interp.invoke()
        predictions = interp.get_tensor(out_details[0]['index'])[0]

        pred_idx = np.argmax(predictions)
        disease = class_labels[pred_idx]
        conf = predictions[pred_idx] * 100

        st.markdown("### 🧪 Diagnosis Result")
        
        if disease == "Healthy":
            st.success(f"✅ **{disease}**")
        elif disease == "Panama Disease":
            st.error(f"🚨 **{disease}**")
        else:
            st.warning(f"⚠️ **{disease}**")
        
        st.metric("Confidence Level", f"{conf:.1f}%")

    st.markdown("---")

    # treatment recommendations - researched from agri extension docs
    treatments = {
        "Cordana": {
            "severity": "warning",
            "emoji": "⚠️",
            "actions": [
                "🔪 Remove and destroy infected leaf spots immediately",
                "🧪 Apply copper-based fungicide (follow local guidelines)",
                "💧 Reduce leaf wetness - avoid overhead irrigation",
                "🌱 Improve plant spacing for better air circulation"
            ],
            "prevention": "Regular monitoring and early removal of infected leaves"
        },
        "Healthy": {
            "severity": "success",
            "emoji": "✅",
            "actions": [
                "✅ Continue current care practices",
                "👀 Monitor weekly for early disease signs",
                "🌿 Maintain proper nutrition (N-P-K balance)",
                "💦 Ensure adequate but not excessive watering"
            ],
            "prevention": "Preventive fungicide application during wet seasons"
        },
        "Panama Disease": {
            "severity": "error",
            "emoji": "🚨",
            "actions": [
                "⚠️ ISOLATE affected plants immediately",
                "🚫 DO NOT replant bananas in this soil for 5+ years",
                "🔥 Remove and burn entire plant (roots included)",
                "🧫 Disinfect tools and footwear to prevent spread",
                "🌾 Consider crop rotation with non-susceptible plants"
            ],
            "prevention": "Use resistant varieties. No cure exists for this disease."
        },
        "Sigatoka": {
            "severity": "warning",
            "emoji": "⚠️",
            "actions": [
                "✂️ Remove heavily infected lower leaves",
                "🧴 Apply systemic fungicide (rotate active ingredients)",
                "🌬️ Improve air circulation - prune dense plantings",
                "📅 Weekly monitoring during humid conditions"
            ],
            "prevention": "Apply fungicide preventively in high-risk periods"
        }
    }
    
    info = treatments[disease]
    
    tab1, tab2, tab3 = st.tabs(["💊 Treatment Plan", "🧠 AI Explanation", "📊 All Predictions"])
    
    with tab1:
        if info["severity"] == "error":
            st.error(f"🚨 **Critical Condition Detected**")
        elif info["severity"] == "warning":
            st.warning(f"⚠️ **Action Required**")
        else:
            st.success(f"✅ **Plant is Healthy**")
        
        st.markdown("#### 📋 Immediate Actions:")
        for action in info["actions"]:
            st.markdown(f"- {action}")
        
        st.markdown("---")
        st.info(f"**🛡️ Prevention Tip:** {info['prevention']}")
    
    with tab2:
        st.markdown("""
        #### 🔍 What the AI Looks For:
        
        **Visual Features Detected:**
        - 🎨 **Color Changes:** Yellowing, browning, or dark spots on leaves
        - 🔍 **Spot Patterns:** Circular lesions, streaks, or irregular patches  
        - 📐 **Texture Damage:** Necrotic areas, wilting, or structural breakdown
        - 🌈 **Contrast Variations:** Differences between healthy and diseased tissue
        
        ---
        
        #### 🦠 Disease Signatures:
        
        **🔴 Cordana:** Small dark spots with yellow halos  
        **🟡 Sigatoka:** Long parallel streaks turning brown/black  
        **🔴 Panama Disease:** Yellowing from leaf edges, wilting  
        **🟢 Healthy:** Uniform green color, no visible damage  
        
        ---
        
        *⚠️ Note: This is an AI prediction tool. For critical agricultural decisions, please consult a certified plant pathologist or agricultural extension officer.*
        """)
    
    with tab3:
        st.markdown("#### 📊 Confidence Scores Across All Classes:")
        st.markdown("---")
        for i, lbl in enumerate(class_labels):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(float(predictions[i]), text=f"**{lbl}**")
            with col_b:
                st.markdown(f"**{predictions[i]*100:.1f}%**")

else:
    st.info("👆 Please upload a banana leaf image to begin diagnosis")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 How to Use:
    
    1. **Take a Photo** of a banana leaf (close-up, good lighting)
    2. **Upload** the image using the button above
    3. **Get Instant Results** with treatment recommendations
    4. **Follow the Action Plan** to protect your crop
    
    ---
    
    ### ✨ Features:
    
    - ⚡ **Instant Detection** - Results in under 1 second
    - 📱 **Mobile-Friendly** - Works on low-end smartphones
    - 🌐 **Offline Ready** - No internet required after download
    - 🎯 **Action-Oriented** - Clear steps for farmers to follow
    """)
    
