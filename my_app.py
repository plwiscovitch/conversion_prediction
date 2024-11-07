import streamlit as st
import pickle
import pandas as pd

# load model
with open("predict_conversion_final.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# set up title and instructions
st.title("Customer Conversion Prediction")
st.write("Enter the following details to get a prediction on conversion likelihood.")


# define input fields
def get_user_input():
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    income = st.number_input("Income", min_value=0)
    ad_spend = st.number_input("Ad Spend", min_value=0.0, step=0.01)
    click_through_rate = st.number_input("Click Through Rate", min_value=0.0, step=0.01)
    conversion_rate = st.number_input("Conversion Rate", min_value=0.0, step=0.01)
    website_visits = st.number_input("Website Visits", min_value=0)
    pages_per_visit = st.number_input("Pages Per Visit", min_value=0.0, step=0.1)
    time_on_site = st.number_input("Time on Site", min_value=0.0, step=0.1)
    email_opens = st.number_input("Email Opens", min_value=0)
    email_clicks = st.number_input("Email Clicks", min_value=0)
    previous_purchases = st.number_input("Previous Purchases", min_value=0)
    loyalty_points = st.number_input("Loyalty Points", min_value=0)
    campaign_type_awareness = st.selectbox("Campaign Type - Awareness", [0, 1])
    campaign_type_consideration = st.selectbox("Campaign Type - Consideration", [0, 1])
    campaign_type_conversion = st.selectbox("Campaign Type - Conversion", [0, 1])
    campaign_type_retention = st.selectbox("Campaign Type - Retention", [0, 1])

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

# get prediction
if st.button("Predict Conversion"):
    prediction = model.predict(input_df)
    st.write("Conversion prediction:", "Yes" if prediction[0] == 1 else "No")


