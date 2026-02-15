import streamlit as st #AnjaliTiwari
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ID3 Tree Visualizer")

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col)) * math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum((len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target]) for v in vals)
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]

    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}

    for val in df[best].unique():
        sub_df = df[df[best] == val]
        new_attrs = [a for a in attrs if a != best]
        tree[best][val] = id3(sub_df, target, new_attrs)
    return tree

st.title("ID3 Decision Tree Classifier")
st.info("This app builds an ID3 tree and visualizes feature importance using Information Gain.")

st.sidebar.header("Configuration")
data_option = st.sidebar.selectbox("Dataset Source", ["Synthetic Tennis Data", "Upload CSV"])

if data_option == "Synthetic Tennis Data":
    data_dict = {
        "Outlook": ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'overcast', 'rain', 'overcast', 'overcast', 'rain', 'sunny'],
        "Humidity": ['high', 'normal', 'high', 'normal', 'high', 'high', 'normal', 'normal', 'normal', 'normal', 'normal', 'high', 'high', 'normal'],
        "Wind": ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
        "PlayTennis": ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data_dict)
else:
    file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

target_col = st.sidebar.selectbox("Select Target Label", df.columns, index=len(df.columns)-1)
features = [c for c in df.columns if c != target_col]

if st.button("Generate Tree & Analyze Features"):
    # 1. Generate Tree
    tree = id3(df, target_col, features)

    # 2. Calculate Information Gain for Visualization
    gains = {feat: info_gain(df, feat, target_col) for feat in features}
    gain_df = pd.DataFrame(list(gains.items()), columns=['Feature', 'Information Gain']).sort_values(by='Information Gain', ascending=False)

    # 3. Visualization Section
    st.subheader("Feature Importance (Information Gain)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Information Gain', y='Feature', data=gain_df, palette='viridis', ax=ax)
    ax.set_title("How the ID3 Algorithm Prioritized Features")
    st.pyplot(fig)

    st.subheader("Decision Logic (JSON)")
    st.json(tree)

    # 4. Target Distribution Plot
    st.subheader("Target Variable Distribution")
    fig2, ax2 = plt.subplots()
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['#ff9999','#66b3ff'])
    ax2.set_ylabel('')
    st.pyplot(fig2)

st.sidebar.divider()
st.sidebar.write("Algorithm: ID3")
st.sidebar.write("Metric: Information Gain")
