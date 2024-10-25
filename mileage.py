import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Optional file uploader for litres of gas data
uploaded_litres_file = st.file_uploader("(Optional) Please upload another Excel file with litres of gas for each truck unit for your selected month to include MPG. (make sure the name of the column name for truck unit number is 'Unit' and for litres is 'Total Qty')", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', skiprows=2)
    trailer_data = trailer_data[trailer_data['UNIT NUMBER'].notna()]
    trailer_data = trailer_data[(trailer_data['Terminal'].notna()) & (trailer_data['Wide'] != 'Texas') & (trailer_data['Terminal'] != 'Texas')]
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()

# Load litres of gas data if a file is uploaded
if uploaded_litres_file:
    litres_data = pd.read_excel(uploaded_litres_file)
    litres_data['Gallons'] = litres_data['Total Qty'] * 0.264172  # Convert litres to gallons

# Filter the relevant date columns
date_columns = ['AUG 1-31', 'Sept 1-30.']
selected_date_column = st.selectbox("Select Date Column", date_columns)

# Function to calculate Target % based on the provided logic
def calculate_target_percentage(row, date_column):
    if row['Type'] == 'Single' and row[date_column] > 10:
        return (row[date_column] / 12000) * 100
    elif row['Type'] == 'Team' and row[date_column] > 10:
        return (row[date_column] / 20000) * 100
    else:
        return None

# Apply the Target % calculation to the relevant date column selected by the user
trailer_data['Target %'] = trailer_data.apply(lambda row: calculate_target_percentage(row, selected_date_column), axis=1)

# If litres data is uploaded, calculate MPG and merge it with trailer data
if uploaded_litres_file:
    litres_data.rename(columns={'Unit': 'UNIT NUMBER'}, inplace=True)
    trailer_data = pd.merge(trailer_data, litres_data[['UNIT NUMBER', 'Gallons']], on='UNIT NUMBER', how='left')
    trailer_data['MPG'] = trailer_data.apply(lambda row: row[selected_date_column] / row['Gallons'] if row['Gallons'] > 0 else None, axis=1)

# Remove duplicates in terminals and standardize names
trailer_data['Terminal'] = trailer_data['Terminal'].replace({'Winnipeg ': 'Winnipeg'})

# Create filters with "All" option and multiple selection enabled
terminals = ['All'] + sorted(trailer_data['Terminal'].unique())
terminal = st.multiselect("Select Terminal", terminals, default='All')

types = ['All'] + sorted(trailer_data['Type'].unique())
type_filter = st.multiselect("Select Type", types, default='All')

wides = ['All'] + sorted(trailer_data['Wide'].unique())
wide = st.multiselect("Select Wide (Geographic Region)", wides, default='All')

planner_names = ['All'] + sorted(trailer_data['Planner Name'].unique())
planner_name = st.multiselect("Select Planner Name", planner_names, default='All')

# Filter data based on selections
filtered_data = trailer_data.copy()
if 'All' not in terminal:
    filtered_data = filtered_data[filtered_data['Terminal'].isin(terminal)]
if 'All' not in type_filter:
    filtered_data = filtered_data[filtered_data['Type'].isin(type_filter)]
if 'All' not in wide:
    filtered_data = filtered_data[filtered_data['Wide'].isin(wide)]
if 'All' not in planner_name:
    filtered_data = filtered_data[filtered_data['Planner Name'].isin(planner_name)]

# Display filtered data and visualizations
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data)
    # Define functions for visualizations here
    # Similar to previous function definitions and use of Plotly for visualization
    # Ensure that MPG is included where applicable

# Add download CSV option
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button("Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
