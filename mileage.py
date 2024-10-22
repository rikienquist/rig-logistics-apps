import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for trailer data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    # Load data while skipping rows with missing UNIT NUMBER
    trailer_data = pd.read_excel(uploaded_file, sheet_name="Review Miles sheet")
    trailer_data = trailer_data[trailer_data['UNIT NUMBER'].notna()]  # Remove rows where UNIT NUMBER is empty

    # Ensure the relevant columns are treated as numeric
    trailer_data['UNIT NUMBER'] = trailer_data['UNIT NUMBER'].astype(str)
    trailer_data['Sept 1-30'] = pd.to_numeric(trailer_data['Sept 1-30'], errors='coerce')
else:
    st.warning("Please upload a Trailer Data Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Filter options
terminal = st.multiselect("Select Terminal", ['All', 'Calgary', 'Edmonton', 'Toronto', 'Winnipeg'], default='All')
wide = st.multiselect("Select Wide (Geographic Region)", ['All', 'Canada', 'USA'], default='All')
type_filter = st.multiselect("Select Type", ['All', 'Single', 'Team'], default='All')
planner_name = st.multiselect("Select Planner Name", trailer_data['Planner Name'].unique(), default='All')

# Apply filters based on user input, skip 'All' options in filtering
if 'All' not in terminal:
    trailer_data = trailer_data[trailer_data['Terminal'].isin(terminal)]
if 'All' not in wide:
    trailer_data = trailer_data[trailer_data['Wide'].isin(wide)]
if 'All' not in type_filter:
    trailer_data = trailer_data[trailer_data['Type'].isin(type_filter)]
if 'All' not in planner_name:
    trailer_data = trailer_data[trailer_data['Planner Name'].isin(planner_name)]

# Validate that the data is filtered properly
st.write("### Filtered Trailer Data")
st.write(trailer_data[['Terminal', 'Wide', 'Route', 'Type', 'UNIT NUMBER', 'Sept 1-30']])

# Calculate Target % as per your logic
def calculate_target_percentage(row):
    if row['Type'] == 'Single' and row['Sept 1-30'] > 10:
        return row['Sept 1-30'] / 12000 * 100
    elif row['Type'] == 'Team' and row['Sept 1-30'] > 10:
        return row['Sept 1-30'] / 20000 * 100
    return None

# Apply the Target % calculation
trailer_data['Target %'] = trailer_data.apply(calculate_target_percentage, axis=1)

# Create a bar chart of Target % by Terminal
def create_target_percentage_chart(data):
    fig = px.bar(
        data_frame=data,
        x="Terminal",
        y="Target %",
        color="Type",
        barmode="stack",
        text="Target %",
        title="Target Percentage by Terminal"
    )
    fig.update_layout(xaxis_title="Terminal", yaxis_title="Target %")
    return fig

# Plot the chart if data is available
if not trailer_data.empty:
    st.write("### Target Percentage Visualization")
    fig = create_target_percentage_chart(trailer_data)
    st.plotly_chart(fig)

# Drill-down section
st.write("### Drill Down")
drill_down_option = st.radio("Drill Down to", ['Routes', 'Unit Numbers'], index=0)

# Drill down by route or unit numbers
if drill_down_option == 'Routes':
    route_data = trailer_data.groupby(['Route']).agg({'Target %': 'mean', 'UNIT NUMBER': 'count'}).reset_index()
    route_data.columns = ['Route', 'Avg Target %', 'Number of Units']
    st.write("### Routes Breakdown")
    st.write(route_data)
else:
    unit_data = trailer_data.groupby(['UNIT NUMBER', 'Route']).agg({'Target %': 'mean'}).reset_index()
    st.write("### Unit Numbers Breakdown")
    st.write(unit_data)

# Option to download filtered data
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not trailer_data.empty:
    csv = convert_df(trailer_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_trailer_data.csv', mime='text/csv')
