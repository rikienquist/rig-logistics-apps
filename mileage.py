import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    # Load the specific sheet and filter out rows with missing UNIT NUMBER
    trailer_data = pd.read_excel(uploaded_file, sheet_name="Review Miles sheet")
    trailer_data = trailer_data[trailer_data['UNIT NUMBER'].notna()]  # Remove rows with no Unit Number

    # Convert UNIT NUMBER to string for consistency
    trailer_data['UNIT NUMBER'] = trailer_data['UNIT NUMBER'].astype(str)

    # Ensure Sept 1-30 is numeric
    trailer_data['Sept 1-30'] = pd.to_numeric(trailer_data['Sept 1-30'], errors='coerce')

else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()

# Multiselect filter for terminal, type, wide, and planner
terminal = st.multiselect("Select Terminal", ['Calgary', 'Edmonton', 'Toronto', 'Winnipeg'], default=['Calgary', 'Edmonton', 'Toronto', 'Winnipeg'])
wide = st.multiselect("Select Wide (Geographic Region)", ['Canada', 'USA'], default=['Canada', 'USA'])
type_filter = st.multiselect("Select Type", ['Single', 'Team'], default=['Single', 'Team'])
planner_name = st.multiselect("Select Planner Name", trailer_data['Planner Name'].unique(), default=trailer_data['Planner Name'].unique())

# Apply filters based on user input
filtered_data = trailer_data[
    (trailer_data['Terminal'].isin(terminal)) &
    (trailer_data['Wide'].isin(wide)) &
    (trailer_data['Type'].isin(type_filter)) &
    (trailer_data['Planner Name'].isin(planner_name))
]

# Ensure the data isn't empty after filtering
if filtered_data.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# Define function to calculate Target %
def calculate_target_percentage(row):
    if row['Type'] == 'Single' and row['Sept 1-30'] > 10:
        return row['Sept 1-30'] / 12000 * 100
    elif row['Type'] == 'Team' and row['Sept 1-30'] > 10:
        return row['Sept 1-30'] / 20000 * 100
    return None

# Apply the Target % calculation
filtered_data['Target %'] = filtered_data.apply(calculate_target_percentage, axis=1)

# Create a stacked bar chart of Target % by Terminal and Type (Single/Team)
def create_target_percentage_chart(data):
    fig = px.bar(
        data_frame=data,
        x="Terminal",
        y="Target %",
        color="Type",
        barmode="stack",  # Stacked bar chart for Single and Team
        text="Target %",
        title="Target Percentage by Terminal"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title="Terminal", yaxis_title="Avg Target %", uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

# Plot the chart
if not filtered_data.empty:
    st.write("### Filtered Trailer Data")
    st.write(filtered_data[['Terminal', 'Wide', 'Route', 'Type', 'UNIT NUMBER', 'Sept 1-30', 'Target %']])

    st.write("### Target Percentage Visualization")
    fig = create_target_percentage_chart(filtered_data)
    st.plotly_chart(fig)

# Drill-down section
st.write("### Drill Down")
drill_down_option = st.radio("Drill Down to", ['Routes', 'Unit Numbers'], index=0)

# Drill down by route or unit numbers
if drill_down_option == 'Routes':
    route_data = filtered_data.groupby(['Route']).agg({'Target %': 'mean', 'UNIT NUMBER': 'count'}).reset_index()
    route_data.columns = ['Route', 'Avg Target %', 'Number of Units']
    st.write("### Routes Breakdown")
    st.write(route_data)
else:
    unit_data = filtered_data.groupby(['UNIT NUMBER', 'Route']).agg({'Target %': 'mean'}).reset_index()
    st.write("### Unit Numbers Breakdown")
    st.write(unit_data)

# Option to download filtered data
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
