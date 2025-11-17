"""
Basic Streamlit Test App
Run with: streamlit run test_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("My First Streamlit App")

# Text
st.write("This is a basic Streamlit app!")

# Header
st.header("Section 1: Text & Formatting")
st.subheader("This is a subheader")
st.text("This is plain text")
st.markdown("**This is bold markdown**")
st.code("print('Hello World')", language='python')

# Divider
st.divider()

# Section 2: Interactive Widgets
st.header("Section 2: Interactive Widgets")

# Slider
number = st.slider("Pick a number", 0, 100, 50)
st.write(f"You picked: {number}")

# Text input
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")

# Button
if st.button("Click me!"):
    st.success("Button clicked!")
    st.balloons()  # Celebration animation!

# Divider
st.divider()

# Section 3: Data Display
st.header("Section 3: Data Display")

# Dataframe
df = pd.DataFrame({
    'Column A': [1, 2, 3, 4],
    'Column B': [10, 20, 30, 40]
})

st.dataframe(df)

# Chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

# Divider
st.divider()

# Section 4: Columns
st.header("Section 4: Layout - Columns")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Metric 1", "100", "+10%")

with col2:
    st.metric("Metric 2", "200", "-5%")

with col3:
    st.metric("Metric 3", "300", "+15%")

# Sidebar
st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar!")
option = st.sidebar.selectbox(
    "Choose an option",
    ["Option 1", "Option 2", "Option 3"]
)
st.sidebar.write(f"You selected: {option}")