import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', skiprows=2)
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()

# Filter the relevant date columns
date_columns = ['AUG 1-31', 'Sept 1-30.']

# Remove any rows with NaN values in important columns
trailer_data.dropna(subset=['Terminal', 'Type', 'Wide', 'Planner Name'], inplace=True)

# Remove duplicates (like duplicate Winnipeg) and unwanted values (like "Texas")
trailer_data = trailer_data[trailer_data['Terminal'] != 'Texas']
trailer_data['Terminal'] = trailer_data['Terminal'].replace('Winnipeg', 'Winnipeg').drop_duplicates()

# Function to create the Target % based on the provided logic
def calculate_target_percentage(row, date_column):
    if row['Type'] == 'Single' and row[date_column] > 10:
        return (row[date_column] / 12000) * 100
    elif row['Type'] == 'Team' and row[date_column] > 10:
        return (row[date_column] / 20000) * 100
    else:
        return None

# Apply the Target % calculation to the relevant date column selected by the user
selected_date_column = st.selectbox("Select Date Column", date_columns)
trailer_data['Target %'] = trailer_data.apply(lambda row: calculate_target_percentage(row, selected_date_column), axis=1)

# Create filters with "All" option and multiple selection enabled
terminals = ['All'] + sorted(trailer_data['Terminal'].unique())
terminal = st.multiselect("Select Terminal", [t for t in terminals if pd.notna(t)], default='All')

types = ['All'] + trailer_data['Type'].unique().tolist()
type_filter = st.multiselect("Select Type", [t for t in types if pd.notna(t)], default='All')

wides = ['All'] + trailer_data['Wide'].unique().tolist()
wide = st.multiselect("Select Wide (Geographic Region)", [w for w in wides if pd.notna(w) and w != "Texas"], default='All')

planner_names = ['All'] + trailer_data['Planner Name'].unique().tolist()
planner_name = st.multiselect("Select Planner Name", [p for p in planner_names if pd.notna(p)], default='All')

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

# Calculate average Target % and count units
avg_target_percentage = filtered_data['Target %'].mean()
unit_count = filtered_data['UNIT NUMBER'].nunique()

# Define function to create a stacked bar chart (Target %)
def create_target_percentage_chart(data):
    avg_target_per_terminal = data.groupby(['Terminal', 'Type'])['Target %'].mean().reset_index()
    unit_count_per_terminal = data.groupby(['Terminal', 'Type'])['UNIT NUMBER'].nunique().reset_index()
    merged_data = pd.merge(avg_target_per_terminal, unit_count_per_terminal, on=['Terminal', 'Type'])

    # Create the stacked bar chart
    fig = go.Figure()
    for type_value in merged_data['Type'].unique():
        filtered_type_data = merged_data[merged_data['Type'] == type_value]
        fig.add_trace(go.Bar(
            x=filtered_type_data['Terminal'],
            y=filtered_type_data['Target %'],
            text=[f"Avg Target: {row['Target %']:.2f}%<br>Units: {row['UNIT NUMBER']}" for i, row in filtered_type_data.iterrows()],
            textposition='auto',
            name=type_value,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Target Percentage by Terminal",
        xaxis_title="Terminal",
        yaxis_title="Avg Target %",
        barmode='stack'
    )
    return fig

# Display filtered data
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data)

    # Display target percentage visualization
    st.write("### Target Percentage Visualization")
    fig = create_target_percentage_chart(filtered_data)
    st.plotly_chart(fig)

    # Drill down to routes and unit numbers
    st.write("### Drill Down")
    drilldown_level = st.radio("Drill Down to", ['Routes', 'Unit Numbers'])

    if drilldown_level == 'Routes':
        route_data = filtered_data.groupby(['Route'])['Target %'].mean().reset_index()
        route_unit_count = filtered_data.groupby(['Route'])['UNIT NUMBER'].nunique().reset_index()
        merged_route_data = pd.merge(route_data, route_unit_count, on='Route')
        st.write("Routes Breakdown", merged_route_data)
    elif drilldown_level == 'Unit Numbers':
        selected_route = st.selectbox("Select Route to Drill Down", filtered_data['Route'].unique())
        unit_data = filtered_data[filtered_data['Route'] == selected_route].groupby(['UNIT NUMBER'])['Target %'].mean().reset_index()
        st.write("Unit Numbers Breakdown", unit_data)
else:
    st.warning("No data available for the selected filters.
