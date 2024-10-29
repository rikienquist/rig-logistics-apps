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
    # Filter out rows where 'UNIT NUMBER' is NaN or missing, and remove rows where 'Terminal' is empty, or 'Wide' or 'Terminal' is 'Texas'
    trailer_data = trailer_data[trailer_data['UNIT NUMBER'].notna()]
    trailer_data = trailer_data[(trailer_data['Terminal'].notna()) & (trailer_data['Wide'] != 'Texas') & (trailer_data['Terminal'] != 'Texas')]
else:
    st.warning("Please upload a Mileage Report Excel file to visualize the data.")
    st.stop()

# Filter the relevant date columns
date_columns = ['AUG 1-31', 'Sept 1-30.']

# Function to create the Target % based on the provided logic
def calculate_target_percentage(row, date_column):
    if row['Type'] == 'Single' and row[date_column] > 10:
        return (row[date_column] / 12000) * 100
    elif row['Type'] == 'Team' and row[date_column] > 10:
        return (row[date_column] / 20000) * 100
    else:
        return None

# Select Date Column dynamically
selected_date_column = st.selectbox("Select Date Column", date_columns)

# Dynamically reset and recalculate Target % and Target Achieved based on the selected column
def recalculate_metrics(data, date_column):
    # Recalculate Target %
    data['Target %'] = data.apply(lambda row: calculate_target_percentage(row, date_column), axis=1)

    # Recalculate Target Achieved status dynamically based on the selected date column
    def check_target_achieved(row):
        if row['Type'] == 'Single' and row[date_column] >= 12000:
            return 'Target Achieved'
        elif row['Type'] == 'Team' and row[date_column] >= 20000:
            return 'Target Achieved'
        else:
            return 'Target Not Achieved'

    data['Target Status'] = data.apply(lambda row: check_target_achieved(row), axis=1)
    return data

# Ensure calculations refresh when selecting a new column
filtered_data = recalculate_metrics(trailer_data.copy(), selected_date_column)

# Remove duplicates in terminals (especially for 'Winnipeg')
filtered_data['Terminal'] = filtered_data['Terminal'].replace({'Winnipeg ': 'Winnipeg'})  # Fix spacing issues

# Normalize Planner Name to handle case and space differences
filtered_data['Planner Name'] = filtered_data['Planner Name'].str.strip().str.title()  # Standardize to title case

# Create filters with "All" option and multiple selection enabled
terminals = ['All'] + sorted(filtered_data['Terminal'].unique())
terminal = st.multiselect("Select Terminal", terminals, default='All')

types = ['All'] + sorted(filtered_data['Type'].unique())
type_filter = st.multiselect("Select Type", types, default='All')

wides = ['All'] + sorted(filtered_data['Wide'].unique())
wide = st.multiselect("Select Wide (Geographic Region)", wides, default='All')

planner_names = ['All'] + sorted(filtered_data['Planner Name'].unique())
planner_name = st.multiselect("Select Planner Name", planner_names, default='All')

# Filter data based on selections
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

# Define function to create a stacked bar chart when both Single and Team are selected
def create_stacked_bar_chart(data):
    avg_target_per_terminal_type = data.groupby(['Terminal', 'Type'])['Target %'].mean().reset_index()
    unit_count_per_terminal_type = data.groupby(['Terminal', 'Type'])['UNIT NUMBER'].nunique().reset_index()
    merged_data = pd.merge(avg_target_per_terminal_type, unit_count_per_terminal_type, on=['Terminal', 'Type'])

    fig = px.bar(
        merged_data,
        x='Terminal',
        y='Target %',
        color='Type',
        text=merged_data.apply(lambda row: f"Avg Target: {row['Target %']:.2f}%<br>Units: {row['UNIT NUMBER']}", axis=1),
        barmode='stack',
        title="Target Percentage by Terminal"
    )

    fig.update_layout(
        xaxis_title="Terminal",
        yaxis_title="Avg Target %",
        showlegend=True
    )
    return fig

# Define function to create a pie chart for Target Achieved Percentage
def create_pie_chart(data):
    target_achieved_count = data[data['Target Status'] == 'Target Achieved'].shape[0]
    total_count = data.shape[0]
    target_not_achieved_count = total_count - target_achieved_count

    fig = go.Figure(
        data=[go.Pie(
            labels=['Target Achieved', 'Target Not Achieved'],
            values=[target_achieved_count, target_not_achieved_count],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=['green', 'red'])
        )]
    )
    fig.update_layout(title_text="Target Achieved Percentage")
    return fig

# Display filtered data
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data)

    # Display target percentage visualization (stacked if both types are selected)
    st.write("### Target Percentage Visualization")
    fig = create_stacked_bar_chart(filtered_data)
    st.plotly_chart(fig)

    # Display pie chart for Target Achieved
    st.write("### Target Achieved Percentage")
    pie_fig = create_pie_chart(filtered_data)
    st.plotly_chart(pie_fig)

    # Drill down to routes and unit numbers
    st.write("### Drill Down")
    drilldown_level = st.radio("Drill Down to", ['Routes', 'Unit Numbers'])

    if drilldown_level == 'Routes':
        # Calculate average target %, average miles, and number of units for each route
        route_data = filtered_data.groupby(['Route'])['Target %'].mean().reset_index()
        route_miles_avg = filtered_data.groupby(['Route'])[selected_date_column].mean().reset_index()
        route_unit_count = filtered_data.groupby(['Route'])['UNIT NUMBER'].nunique().reset_index()

        # Merge all route-level calculations and rearrange columns
        merged_route_data = pd.merge(route_data, route_miles_avg, on='Route')
        merged_route_data = pd.merge(merged_route_data, route_unit_count, on='Route')
        merged_route_data.columns = ['Route', 'Average Miles', 'Number of Units', 'Average Target %']

        st.write("Routes Breakdown", merged_route_data)

    elif drilldown_level == 'Unit Numbers':
        selected_route = st.selectbox("Select Route to Drill Down", filtered_data['Route'].unique())
        
        # Filter data for the selected route and rearrange columns
        unit_data = filtered_data[filtered_data['Route'] == selected_route][['UNIT NUMBER', selected_date_column, 'Target %', 'Comments']]
        unit_data.columns = ['Unit Number', 'Number of Miles', 'Target %', 'Comments']
        
        st.write("Unit Numbers Breakdown", unit_data)
else:
    st.warning("No data available for the selected filters.")

# Add an option to download filtered data as CSV
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
