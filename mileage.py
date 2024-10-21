import pandas as pd
import streamlit as st
import plotly.express as px
import re

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    # Read in the Excel file and ensure the correct sheet is used
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', header=2)  # Starting at row 3 (0-indexed means header=2)
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Helper function to filter columns that contain month names
def filter_date_columns(columns):
    month_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', re.IGNORECASE)
    return [col for col in columns if month_pattern.search(col)]

# Filter the date columns
date_columns = st.multiselect("Select Date Columns", filter_date_columns(trailer_data.columns))

# Restrict terminal options to Calgary, Edmonton, Toronto, and Winnipeg
valid_terminals = ['Calgary', 'Edmonton', 'Toronto', 'Winnipeg']
terminal = st.selectbox("Select Terminal", valid_terminals)

# Restrict wide options to Canada and USA
wide = st.selectbox("Select Wide (Geographic Region)", ['Canada', 'USA'])

# Restrict type options based on unique types in the dataset
type_filter = st.selectbox("Select Type", trailer_data['Type'].unique())

# Restrict planner name options based on unique planner names in the dataset
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
    filtered_data = filtered_data[date_columns + ['Terminal', 'Type', 'Wide', 'Planner Name', 'Route', 'UNIT NUMBER']]

# Define function to create a stacked bar chart with drill-down
def create_target_percentage_chart(data):
    fig = px.bar(
        data_frame=data,
        x="Terminal",  # Group by Terminal
        y="Sept 1-30",  # Replace with the actual column for percentage
        color="Type",  # Color by Single/Team
        barmode="stack",  # Stacked bar chart
        hover_data=["Route", "UNIT NUMBER"],  # Add hover data for Route and Unit Number
        title="Target Percentage by Terminal"
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Terminal",
        yaxis_title="Target %",
        hovermode="closest"
    )
    
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
