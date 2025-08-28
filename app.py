import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# ------------------ USER LOGIN ------------------
USER_CREDENTIALS = {
    "customer1@example.com": "password123",
    "customer2@example.com": "secure456"
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# Login form
if not st.session_state.logged_in:
    st.title("🔐 Customer Dashboard Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.success(f"✅ Login successful! Welcome {email}")
        else:
            st.error("❌ Invalid email or password")

    # Stop execution until login
    if not st.session_state.logged_in:
        st.stop()

# ------------------ DASHBOARD ------------------
st.title(f"📊 Customer Data Dashboard - {st.session_state.user_email}")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ------------------ Basic Stats ------------------
    st.write("### Dataset Info")
    st.write(df.describe(include='all'))

    # ------------------ Column selection ------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    st.write("### Visualization Options")
    chart_type = st.selectbox("Select chart type", ["Bar Chart", "Correlation Heatmap", "Scatter Plot"])

    if chart_type == "Bar Chart":
        if numeric_cols:
            col_to_plot = st.selectbox("Select numeric column for Bar Chart", numeric_cols)
            st.bar_chart(df[col_to_plot])
        else:
            st.info("⚠️ No numeric columns available for Bar Chart.")

    elif chart_type == "Correlation Heatmap":
        if numeric_cols:
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("⚠️ No numeric columns available for Correlation Heatmap.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X axis", numeric_cols, index=0)
            y_col = st.selectbox("Select Y axis", numeric_cols, index=1)
            hue_col = None
            if categorical_cols:
                hue_col = st.selectbox("Optional: Select categorical column for color", [None]+categorical_cols)
            fig, ax = plt.subplots()
            if hue_col:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        else:
            st.info("⚠️ Need at least 2 numeric columns for Scatter Plot.")

    # ------------------ Download as Excel ------------------
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
else:
    st.info("📁 Upload a CSV file to start analyzing.")

# ------------------ LOGOUT BUTTON ------------------
if st.session_state.logged_in:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.experimental_rerun()
