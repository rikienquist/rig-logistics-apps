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
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', skiprows=1)
    
    # Handle duplicate columns (keep last occurrence for newest data)
    trailer_data = trailer_data.loc[:, ~trailer_data.columns.duplicated(keep='last')]

    # Remove rows where 'UNIT NUMBER' or 'Terminal' is missing
    trailer_data = trailer_data.dropna(subset=['UNIT NUMBER', 'Terminal'])

    # Standardize 'Terminal', 'Wide', and 'Type' columns
    for col in ['Terminal', 'Wide', 'Type']:
        if col in trailer_data.columns:
            trailer_data[col] = trailer_data[col].astype(str).str.strip()
            if col in ['Terminal', 'Wide']:
                trailer_data[col] = trailer_data[col].str.title()  # Convert to title case (e.g., "Edmonton")
            if col == 'Type':
                trailer_data[col] = trailer_data[col].str.upper()  # Convert to uppercase (e.g., "SINGLE")
    
    # Remove invalid rows where 'Type' is NaN
    trailer_data = trailer_data[trailer_data['Type'].notna()]

    # Standardize 'Planner Name' column
    trailer_data['Planner Name'] = trailer_data['Planner Name'].fillna("").astype(str).str.strip().str.title()

else:
    st.warning("Please upload a Mileage Report Excel file to visualize the data.")
    st.stop()

# Define relevant date columns explicitly
date_columns = ['AUG 1-31', 'Sept 1-30.', 'OCT 1-31', 'NOV 1-30', 'DEC 1-31', 'Jan 1-31', 'FEB 1-28']

# Identify and select the most recent (rightmost) columns when duplicates exist
available_date_columns = []
for col in date_columns:
    matching_cols = [c for c in trailer_data.columns if col in c]
    if matching_cols:
        available_date_columns.append(matching_cols[-1])  # Select the latest version

if not available_date_columns:
    st.error("No valid date columns found in the uploaded file.")
    st.stop()

# Select Date Column dynamically
selected_date_column = st.selectbox("Select Date Column", available_date_columns)

# Function to calculate the Target %
def calculate_target_percentage(row, date_column):
    if row['Type'] == 'SINGLE' and row[date_column] > 10:
        return (row[date_column] / 12000) * 100
    elif row['Type'] == 'TEAM' and row[date_column] > 10:
        return (row[date_column] / 20000) * 100
    else:
        return None

# Dynamically reset and recalculate Target % and Target Achieved based on the selected column
def recalculate_metrics(data, date_column):
    # Recalculate Target %
    data['Target %'] = data.apply(lambda row: calculate_target_percentage(row, date_column), axis=1)

    # Recalculate Target Achieved status dynamically based on the selected date column
    def check_target_achieved(row):
        if row['Type'] == 'SINGLE' and row[date_column] >= 12000:
            return 'Target Achieved'
        elif row['Type'] == 'TEAM' and row[date_column] >= 20000:
            return 'Target Achieved'
        else:
            return 'Target Not Achieved'

    data['Target Status'] = data.apply(lambda row: check_target_achieved(row), axis=1)
    return data

# Ensure calculations refresh when selecting a new column
filtered_data = recalculate_metrics(trailer_data.copy(), selected_date_column)

# Remove duplicates in terminals (especially for 'Winnipeg')
filtered_data['Terminal'] = filtered_data['Terminal'].replace({'Winnipeg ': 'Winnipeg'})  # Fix spacing issues

# Create filters with "All" option and multiple selection enabled
terminals = ['All'] + sorted(filtered_data['Terminal'].unique())
terminal = st.multiselect("Select Terminal", terminals, default='All')

types = ['All'] + sorted(filtered_data['Type'].unique())
type_filter = st.multiselect("Select Type", types, default='All')

wides = ['All'] + sorted(filtered_data['Wide'].unique())
wide = st.multiselect("Select Wide (Geographic Region)", wides, default='All')

planner_names = ['All'] + sorted(set(filtered_data['Planner Name']))  # Fix duplicate names
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

# Define function to create a stacked bar chart
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

    # Display target percentage visualization
    st.write("### Target Percentage Visualization")
    fig = create_stacked_bar_chart(filtered_data)
    st.plotly_chart(fig)

    # Display pie chart for Target Achieved
    st.write("### Target Achieved Percentage")
    pie_fig = create_pie_chart(filtered_data)
    st.plotly_chart(pie_fig)

else:
    st.warning("No data available for the selected filters.")

# Add an option to download filtered data as CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
