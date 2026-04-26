import os
import re
import pickle
import torch
import torch.nn as nn
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel

ARTIFACT_DIR = 'gbv_mtl_roberta_model'

# Safety Guidelines Dictionary
SAFETY_PROTOCOLS = {
    'sexual_violence': "### 🚨 IMMEDIATE ACTION REQUIRED\n1. **Medical Help**: Seek immediate medical attention (within 72 hours) for PEP/EC.\n2. **Safe Space**: Contact specialized GBV clinics or safe houses.\n3. **Legal**: Report to authorities if safe to do so.",
    'Physical_violence': "### 🛡️ SAFETY FIRST\n1. **Escape**: Find a secure location immediately.\n2. **Emergency**: Contact local emergency services or police.\n3. **Evidence**: If possible, document injuries once in a safe place.",
    'emotional_violence': "### ❤️ PSYCHOLOGICAL SUPPORT\n1. **Counseling**: Reach out to professional mental health services.\n2. **Support Groups**: You are not alone; connecting with survivors can help.\n3. **Boundaries**: Focus on your mental well-being and safety.",
    'economic_violence': "### ⚖️ FINANCIAL & LEGAL AID\n1. **Resources**: Consult legal aid clinics regarding your rights.\n2. **Independence**: Reach out to social services for economic empowerment programs.",
    'Harmful_Traditional_practice': "### 🌍 COMMUNITY ADVOCACY\n1. **NGO Support**: Seek help from human rights groups specialized in traditional practices.\n2. **Protection**: Legal protections often exist for these specific cases.",
    'Non-GBV': "### ✅ NO IMMEDIATE GBV RISK DETECTED\nThe content does not appear to indicate a high-risk GBV incident. However, stay informed about local support resources."
}

def preprocess_for_roberta(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('#', '')
    return text

class MultiTaskGBVModel(nn.Module):
    def __init__(self, model_name, num_gbv_labels, num_intensity_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Exact structure from the training code to ensure state_dict matches
        self.gbv_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_gbv_labels)
        )

        self.intensity_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_intensity_labels)
        )

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        gbv_logits = self.gbv_classifier(cls_token)
        intensity_logits = self.intensity_classifier(cls_token)
        
        return gbv_logits, intensity_logits

@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, 'label_mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
    model = MultiTaskGBVModel(mappings['model_name'], len(mappings['gbv_labels']), 3)
    model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, 'multitask_roberta_model.pt'), map_location='cpu'))
    model.eval()
    return tokenizer, model, mappings

# --- App Layout ---
st.set_page_config(page_title='GBV Detection System', layout='wide', page_icon='🛡️')

with st.sidebar:
    st.title("Capstone Project")
    st.info("**GBV Multi-Task Classifier**")
    st.divider()
    st.markdown("**Model Details:**\n- CardiffNLP RoBERTa")

tokenizer, model, mappings = load_artifacts()

st.title('🛡️ Gender-Based Violence Detection & Support')
st.markdown("--- ")

tab1, tab2 = st.tabs(['🔍 Case Analysis', '📁 Batch Processing'])

with tab1:
    st.subheader("Analyze Incident Description")
    user_input = st.text_area('Paste text here:', height=150)
    
    if st.button('Classify & Generate Protocol'):
        if user_input.strip():
            clean_text = preprocess_for_roberta(user_input)
            encoded = tokenizer(clean_text, return_tensors='pt', truncation=True, padding='max_length', max_length=mappings['max_len'])
            
            with torch.no_grad():
                g_log, i_log = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            
            g_probs = torch.softmax(g_log, dim=1)
            g_idx = torch.argmax(g_probs).item()
            conf = g_probs[0][g_idx].item()
            label = mappings['id2label'][g_idx]
            i_idx = torch.argmax(i_log).item()

            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.success("### Prediction Results")
                st.metric("Detected GBV Type", label)
                st.write(f"**Confidence:** {conf:.2%}")
                st.progress(conf)
                st.write(f"**Intensity:** {mappings['intensity_id2label'][i_idx]}")

            with col2:
                st.warning("### Recommended Safety Protocol")
                st.markdown(SAFETY_PROTOCOLS.get(label, 'Seek counseling.'))

with tab2:
    st.subheader("Bulk Analysis")
    uploaded_file = st.file_uploader('Choose a file', type=['csv', 'xlsx'])
    if uploaded_file:
        df_batch = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        if st.button('🚀 Run Batch Process'):
            text_col = df_batch.columns[0]
            results = []
            for text in df_batch[text_col]:
                clean = preprocess_for_roberta(text)
                enc = tokenizer(clean, return_tensors='pt', truncation=True, padding='max_length', max_length=mappings['max_len'])
                with torch.no_grad():
                    g_log, _ = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                results.append(mappings['id2label'][torch.argmax(g_log).item()])
            df_batch['GBV_Prediction'] = results
            st.dataframe(df_batch)
            st.download_button('📥 Download Results', df_batch.to_csv(index=False), 'results.csv', 'text/csv')
