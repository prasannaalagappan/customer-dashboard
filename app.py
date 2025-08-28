import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------ USER LOGIN ------------------
USER_CREDENTIALS = {
    "prasanna1@example.com": "password123",
    "prasanna2@example.com": "secure456"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

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
    if not st.session_state.logged_in:
        st.stop()

# ------------------ DASHBOARD ------------------
st.title(f"📊 Advanced Customer Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ------------------ Identify Numeric Columns ------------------
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    numeric_cols = numeric_df.dropna(axis=1, how='all').columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ------------------ KPI Summary ------------------
    st.write("### 📌 Key Metrics")
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        for idx, col_name in enumerate(numeric_cols[:3]):
            col = [col1, col2, col3][idx]
            col.metric(
                label=col_name,
                value=round(numeric_df[col_name].sum(skipna=True), 2),
                delta=f"Mean: {round(numeric_df[col_name].mean(skipna=True), 2)}"
            )
    else:
        st.info("⚠️ No numeric columns available for KPIs.")

    # ------------------ Filters ------------------
    st.write("### 🔍 Filter Data")
    filtered_df = df.copy()
    for col in numeric_cols:
        col_series = numeric_df[col].dropna()
        if not col_series.empty:
            min_val, max_val = float(col_series.min()), float(col_series.max())
            selected_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(numeric_df[col] >= selected_range[0]) & (numeric_df[col] <= selected_range[1])]
    for col in categorical_cols:
        if df[col].nunique() <= 20:
            selected_vals = st.multiselect(f"Filter {col}", options=df[col].unique(), default=list(df[col].unique()))
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.write("### Filtered Data Preview")
    st.dataframe(filtered_df.head())

    # ------------------ Visualizations ------------------
    st.write("### 📈 Visualization Options")
    chart_type = st.selectbox("Select chart type", ["Bar Chart", "Correlation Heatmap", "Scatter Plot", "Boxplot", "Pie Chart"])

    safe_numeric_cols = [col for col in numeric_cols if filtered_df[col].notna().any()]
    safe_categorical_cols = [col for col in categorical_cols if filtered_df[col].notna().any()]

    if chart_type == "Bar Chart" and safe_numeric_cols:
        col_to_plot = st.selectbox("Select numeric column for Bar Chart", safe_numeric_cols)
        st.bar_chart(filtered_df[col_to_plot].dropna())
    elif chart_type == "Correlation Heatmap" and len(safe_numeric_cols) >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(filtered_df[safe_numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    elif chart_type == "Scatter Plot" and len(safe_numeric_cols) >= 2:
        x_col = st.selectbox("X axis", safe_numeric_cols, index=0)
        y_col = st.selectbox("Y axis", safe_numeric_cols, index=1)
        hue_col = None
        if safe_categorical_cols:
            hue_col = st.selectbox("Optional categorical column for color", [None]+safe_categorical_cols)
        fig, ax = plt.subplots()
        if hue_col:
            sns.scatterplot(data=filtered_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        else:
            sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot" and safe_numeric_cols:
        col_to_plot = st.selectbox("Select numeric column for Boxplot", safe_numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(y=filtered_df[col_to_plot].dropna(), ax=ax)
        st.pyplot(fig)
    elif chart_type == "Pie Chart" and safe_categorical_cols:
        col_to_plot = st.selectbox("Select categorical column for Pie Chart", safe_categorical_cols)
        pie_data = filtered_df[col_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
        st.pyplot(fig)

    # ------------------ Predictive Feature ------------------
    st.write("### 🤖 Predict a Numeric Column (Regression)")
    if len(safe_numeric_cols) >= 2:
        target_col = st.selectbox("Select target column to predict", safe_numeric_cols)
        feature_cols = st.multiselect("Select feature columns", [c for c in safe_numeric_cols if c != target_col])
        if st.button("Train & Predict") and feature_cols:
            X = filtered_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            y = filtered_df[target_col].apply(pd.to_numeric, errors='coerce').fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.success(f"Model trained! Mean Squared Error: {round(mse, 2)}")
            pred_df = X_test.copy()
            pred_df[f"{target_col}_predicted"] = predictions
            st.dataframe(pred_df.head())
    else:
        st.info("⚠️ Not enough numeric columns for regression prediction.")

    # ------------------ Download Filtered Data ------------------
    st.write("### 📥 Download Filtered Data as Excel")
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="FilteredData")
    towrite.seek(0)
    st.download_button(
        label="Download Filtered Data",
        data=towrite,
        file_name="customer_data_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("📁 Upload a CSV file to start analyzing.")

# ------------------ LOGOUT ------------------
if st.session_state.logged_in:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.experimental_rerun()
