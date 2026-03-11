# app.py - Complete Streamlit App
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

# Page config
st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="wide"
)

# Title with creator name
st.title("🏦 Banking Customer Intent Classifier")
st.markdown(f"<h5 style='text-align: right; color: #888;'>Created by: Chaudhary Hadi</h5>", unsafe_allow_html=True)
st.markdown("---")

# Load model (cached)
@st.cache_resource
def load_model():
    # Download from Hugging Face hub
    model_path = snapshot_download("chhadi14/banking77-bert-finetuned")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Label names (Banking77)
    label_names = [
        'activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support',
        'automatic_top_up', 'balance_not_updated_after_bank_transfer',
        'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed',
        'cancel_transfer', 'card_about_to_expire', 'card_acceptance',
        'card_arrival', 'card_delivery_estimate', 'card_linking', 'card_not_working',
        'card_payment_fee_charged', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate',
        'card_swallowed', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised',
        'change_pin', 'compromised_card', 'contactless_not_working', 'country_support',
        'declined_card_payment', 'declined_cash_withdrawal', 'declined_transfer',
        'direct_debit_payment_not_recognised', 'disposable_card_limits', 'edit_personal_details',
        'exchange_charge', 'exchange_rate', 'exchange_via_app', 'extra_charge_on_statement',
        'failed_transfer', 'fiat_currency_support', 'get_disposable_virtual_card',
        'get_physical_card', 'getting_spare_card', 'getting_virtual_card', 'lost_or_stolen_card',
        'lost_or_stolen_phone', 'order_physical_card', 'passcode_forgotten', 'pending_card_payment',
        'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked',
        'receiving_money', 'Refund_not_showing_up', 'request_refund', 'reverted_card_payment?',
        'supported_cards_and_currencies', 'terminate_account', 'top_up_by_bank_transfer_charge',
        'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits',
        'top_up_reverted', 'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged',
        'transfer_into_account', 'transfer_not_received_by_recipient', 'transfer_timing', 'unable_to_verify_identity',
        'verify_my_identity', 'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working',
        'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received',
        'wrong_exchange_rate_for_cash_withdrawal'
    ]
    
    return tokenizer, model, label_names

# Load with spinner
with st.spinner("Loading model... Please wait..."):
    tokenizer, model, label_names = load_model()

# Sidebar with creator info
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Below this threshold, query will be flagged for human review"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info(f"• **Classes:** 77 intents\n• **Accuracy:** 93%\n• **Model:** BERT-base")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    **Chaudhary Hadi**  
    ML Engineer & AI Specialist
    
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chhadi14)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://x.com/ChaudharyHadi27)
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter your query")
    
    # Example queries
    example = st.selectbox(
        "Try an example:",
        ["", "I lost my credit card", "Can you increase my limit?", 
         "How do I activate my card?", "My ATM card is not working",
         "What's the interest rate?", "I forgot my PIN"]
    )
    
    # Text input
    if example:
        query = st.text_area("Or type your own:", value=example, height=100)
    else:
        query = st.text_area("Type your banking query here:", height=100)
    
    # Classify button
    if st.button("🔍 Classify Intent", type="primary"):
        if query:
            with st.spinner("Analyzing..."):
                # Tokenize and predict
                inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                    confidence, pred = torch.max(probs, dim=-1)
                
                intent = label_names[pred.item()]
                confidence_score = confidence.item()
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Confidence meter
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Predicted Intent", intent.replace('_', ' ').title())
                with col_b:
                    st.metric("Confidence", f"{confidence_score:.1%}")
                
                # Color-coded confidence
                if confidence_score >= confidence_threshold:
                    st.success("✅ High Confidence - Can be automated")
                else:
                    st.error("🔴 Low Confidence - Needs human review")
                
                # Show top 3 predictions
                st.markdown("### 🔍 Top 3 Predictions")
                top3 = torch.topk(probs, 3)
                
                for i in range(3):
                    intent_name = label_names[top3.indices[i].item()].replace('_', ' ').title()
                    prob_value = top3.values[i].item()
                    st.progress(prob_value, text=f"{i+1}. {intent_name}: {prob_value:.1%}")
        else:
            st.warning("Please enter a query")

with col2:
    st.subheader("ℹ️ About")
    st.markdown("""
    This app classifies banking customer queries into **77 different intents**.
    
    **How it works:**
    1. Enter your question
    2. Model analyzes the text
    3. Get instant intent classification
    4. Confidence score shows reliability
    
    **Accuracy:** 93% on test set
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Sample Intents")
    st.info("""
    • activate_my_card
    • lost_or_stolen_card
    • pin_blocked
    • card_not_working
    • transfer_timing
    • exchange_rate
    • top_up_limits
    """)

# Footer with your name
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>Made with ❤️ by <strong>Chaudhary Hadi</strong> for Banking Customer Support</p>
    <p style='font-size: 0.8em; color: #888;'>© 2026 Chaudhary Hadi. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)