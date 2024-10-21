import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    # Load the data and specify that the column names are in the third row (index 2 in Python)
    trailer_data = pd.read_excel(uploaded_file, header=2)
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Select columns to filter
date_columns = st.multiselect("Select Date Columns", trailer_data.columns)
terminal = st.selectbox("Select Terminal", trailer_data['Terminal'].unique())
type_filter = st.selectbox("Select Type", trailer_data['Type'].unique())
wide = st.selectbox("Select Wide (Geographic Region)", trailer_data['Wide'].unique())
planner_name = st.selectbox("Select Planner Name", trailer_data['Planner Name'].unique())

# Filter data based on selections
filtered_data = trailer_data[
    (trailer_data['Terminal'] == terminal) &
    (trailer_data['Type'] == type_filter) &
    (trailer_data['Wide'] == wide) &
    (trailer_data['Planner Name'] == planner_name)
]

# Filter based on selected date columns
if date_columns:
    filtered_data = filtered_data[date_columns + ['Terminal', 'Type', 'Wide', 'Planner Name']]

# Define function to create a bar chart (Target %)
def create_target_percentage_chart(data):
    fig = px.bar(
        data_frame=data,
        x="Terminal",  # Change based on how you want to visualize
        y="Sept 1-30",  # Example column, adjust as needed
        color="Type",  # Color by Type (Single/Team)
        barmode="group",
        title="Target Percentage by Terminal"
    )
    fig.update_layout(xaxis_title="Terminal", yaxis_title="Target %")
    return fig

# Plot the chart
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data)
    
    st.write("### Target Percentage Visualization")
    fig = create_target_percentage_chart(filtered_data)
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected filters.")

# Add an option to download filtered data as CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
