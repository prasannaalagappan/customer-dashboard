import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# ------------------ USER LOGIN ------------------
# Store login credentials (in real use, fetch from database)
USER_CREDENTIALS = {
    "customer1@example.com": "password123",
    "customer2@example.com": "secure456"
}

# Session state to track login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    st.title("🔐 Customer Dashboard Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid email or password")
    st.stop()

# ------------------ DASHBOARD ------------------
st.title("📊 Customer Data Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Basic stats
    st.write("### Dataset Info")
    st.write(df.describe())

    # Visualization
    st.write("### Visualization")
    st.bar_chart(df.select_dtypes(include="number").iloc[:, 0])

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Download as Excel
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    towrite.seek(0)

    st.download_button(
        label="📥 Download Data as Excel",
        data=towrite,
        file_name="customer_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
