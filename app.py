import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("COVID-19 Data Analysis and Visualization")


@st.cache_data
def load_data():
    confirmed_df = pd.read_csv('covid19_Confirmed_dataset.csv')
    deaths_df = pd.read_csv('covid19_deaths_dataset.csv')
    happiness_df = pd.read_csv('worldwide_happiness_report (1).csv')
    return confirmed_df, deaths_df, happiness_df


@st.cache_data
def process_data(confirmed_df):
    confirmed_df.drop(["Lat", "Long"], axis=1, inplace=True)
    aggregated_df = confirmed_df.groupby("Country/Region").sum()
    return aggregated_df


confirmed_df, deaths_df, happiness_df = load_data()
corona_dataset_aggregated = process_data(confirmed_df)

st.header("Confirmed Cases Over Time")
country = st.selectbox("Select a country", corona_dataset_aggregated.index)

if country:
    st.subheader(f"COVID-19 Confirmed Cases in {country}")
    country_data = corona_dataset_aggregated.loc[country]
    st.line_chart(country_data)

# Ensure data is numeric before calculating max infection rates
corona_dataset_aggregated = corona_dataset_aggregated.apply(pd.to_numeric, errors='coerce')

# Calculate maximum infection rates
countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for c in countries:
    max_infection_rates.append(corona_dataset_aggregated.loc[c].diff().max())
corona_dataset_aggregated["max_infection_rates"] = max_infection_rates
corona_data = pd.DataFrame(corona_dataset_aggregated["max_infection_rates"])

# Clean and prepare happiness data
happiness_df.drop(["Overall rank", "Score", "Generosity", "Perceptions of corruption"], axis=1, inplace=True)
happiness_df.set_index("Country or region", inplace=True)

# Merge data
data = corona_data.join(happiness_df, how="inner")

st.header("Correlation between COVID-19 and Happiness Metrics")
metric = st.selectbox("Select a happiness metric",
                      ["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices"])

if metric:
    st.subheader(f"Correlation between {metric} and Max Infection Rates")
    x = data[metric]
    y = data["max_infection_rates"]

    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=np.log(y), ax=ax)
    sns.regplot(x=x, y=np.log(y), scatter=False, ax=ax)
    st.pyplot(fig)

st.write("Data processing and visualization complete.")
