import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn


# load model
with open("predict_conversion_final.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# set up title and instructions
st.title("Customer Conversion Prediction")
st.header("Enter the following details to determine if conversion will occur.")
st.write("Conversion = Yes means that the consumer will finalize a purchase and Conversion = No means they are not likely to make a purchase.")


# define input fields
def get_user_input():
    age = st.number_input("Age: enter valid age 18 or greater.", min_value=18, max_value=100, step=1)
    income = st.number_input("Annual Income", min_value=0.00, step=0.01)
    ad_spend = st.number_input("Ad Spend: enter cost of ad 100-10000.", min_value=100.00, max_value=10000.00, step=0.01)
    click_through_rate = st.number_input("Click Through Rate: enter value 0.01-0.30.", min_value=0.01, max_value=0.30, step=0.01)
    conversion_rate = st.number_input("Conversion Rate: enter value 0.01-0.20.", min_value=0.01, max_value=0.21, step=0.01)
    website_visits = st.number_input("Website Visits: enter value 0-50.", min_value=0, max_value=50, step=1)
    pages_per_visit = st.number_input("Pages Per Visit: enter value 1-20.", min_value=0.0, max_value=20.0, step=1.0)
    time_on_site = st.number_input("Time on Site: enter value 0.50-15 minutes.", min_value=0.0, step=0.1)
    email_opens = st.number_input("Email Opens: enter value 0-20", min_value=0, max_value=20, step=1)
    email_clicks = st.number_input("Email Clicks: enter value 0-10", min_value=0, max_value=10, step=1)
    previous_purchases = st.number_input("Previous Purchases: enter value 0-10", min_value=0, max_value=10, step=1)
    loyalty_points = st.number_input("Loyalty Points: enter value 0-5000", min_value=0, max_value=5000, step=1)

    st.write("Select the campaign type by entering 1 for the correct campaign type leave all others 0.")
    campaign_type_awareness = st.selectbox("Campaign Type - Awareness: 0=No, 1=Yes", [0, 1])
    campaign_type_consideration = st.selectbox("Campaign Type - Consideration: 0=No, 1=Yes", [0, 1])
    campaign_type_conversion = st.selectbox("Campaign Type - Conversion: 0=No, 1=Yes", [0, 1])
    campaign_type_retention = st.selectbox("Campaign Type - Retention: 0=No, 1=Yes", [0, 1])

    data = {"Age": age,
            "Income": income,
            "AdSpend": ad_spend,
            "ClickThroughRate": click_through_rate,
            "ConversionRate": conversion_rate,
            "WebsiteVisits": website_visits,
            "PagesPerVisit": pages_per_visit,
            "TimeOnSite": time_on_site,
            "EmailOpens": email_opens,
            "EmailClicks": email_clicks,
            "PreviousPurchases": previous_purchases,
            "LoyaltyPoints": loyalty_points,
            "CampaignType_Awareness": campaign_type_awareness,
            "CampaignType_Consideration": campaign_type_consideration,
            "CampaignType_Conversion": campaign_type_conversion,
            "CampaignType_Retention": campaign_type_retention
    }

    features = pd.DataFrame([data])
    return features

# get user input
input_df = get_user_input()

# placeholder for prediction output
prediction_placeholder = st.empty()

# get prediction
if st.button("Predict Conversion"):
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    prediction_text = "Yes, a purchase will be made." if prediction[0] == 1 else "No purchase will be made."
    probability_text = f"Probability of 'Yes': {probabilities[1]:.2%}, Probability of 'No': {probabilities[0]:.2%}"

    # stylize output
    prediction_placeholder.markdown(
        f"<h3 style='color: blue; font-size: 24px;'>Conversion prediction: {prediction_text}</h3>"
        f"<h4>Probability: {probability_text}</h4>",
        unsafe_allow_html=True
    )

