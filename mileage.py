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
    # Read the "Review Miles sheet" sheet and skip the first two rows to use the third row as the header
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', header=2)
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Filter date columns to only include ones with month names
def filter_date_columns(columns):
    # Regex to check for any month name in the column headers
    return [col for col in columns if re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', col, re.IGNORECASE)]

date_columns = filter_date_columns(trailer_data.columns)

# Filter terminal options to only show specified terminals
terminal = st.selectbox("Select Terminal", ['Calgary', 'Edmonton', 'Toronto', 'Winnipeg'])

# Filter wide options to only show Canada and USA
wide = st.selectbox("Select Wide (Geographic Region)", ['Canada', 'USA'])

# Filter by type
type_filter = st.selectbox("Select Type", trailer_data['Type'].unique())

# Filter by planner name
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

# Define function to create a stacked bar chart with drill-downs
def create_drill_down_chart(data):
    fig = px.bar(
        data_frame=data,
        x="Terminal",
        y="Sept 1-30",  # Change based on your target data column
        color="Type",
        hover_data=["Route", "UNIT NUMBER"],  # To show routes and unit numbers
        title="Target Percentage by Terminal",
        barmode="stack"
    )

    fig.update_layout(xaxis_title="Terminal", yaxis_title="Target %")
    return fig

# Plot the chart
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data)
    
    st.write("### Target Percentage Visualization with Drill-Down")
    fig = create_drill_down_chart(filtered_data)
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected filters.")

# Add an option to download filtered data as CSV
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
