import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit app
st.title("Mileage Target Percentages")

# File uploader for mileage data
uploaded_file = st.file_uploader("Upload Mileage Excel File", type=['xlsx'])

# Load trailer data if a file is uploaded
if uploaded_file:
    # Read the 'Review Miles sheet' starting from the correct row
    trailer_data = pd.read_excel(uploaded_file, sheet_name='Review Miles sheet', header=3)  # Adjust header
else:
    st.warning("Please upload a Mileage Data Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Filter only date-related columns (Jan-Dec) and ensure we're only handling string column names
date_columns = [col for col in trailer_data.columns if isinstance(col, str) and any(month in col.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'])]

# Streamlit filters for user to select
selected_date_column = st.selectbox("Select Date Columns", date_columns)
terminal = st.selectbox("Select Terminal", ['Calgary', 'Edmonton', 'Toronto', 'Winnipeg'])
wide = st.selectbox("Select Wide (Geographic Region)", ['Canada', 'USA'])
type_filter = st.selectbox("Select Type", trailer_data['Type'].unique())
planner_name = st.selectbox("Select Planner Name", trailer_data['Planner Name'].unique())

# Filter data based on selections
filtered_data = trailer_data[
    (trailer_data['Terminal'] == terminal) &
    (trailer_data['Type'] == type_filter) &
    (trailer_data['Wide'] == wide) &
    (trailer_data['Planner Name'] == planner_name)
]

# Filter based on selected date columns
if not filtered_data.empty:
    # Only include the necessary columns
    filtered_data = filtered_data[['Terminal', 'Type', 'Wide', 'Planner Name', 'Route', 'UNIT NUMBER', selected_date_column]]

    # Calculate the Target % based on the user selection
    def calculate_target_percentage(row):
        if row['Type'] == 'Single' and row[selected_date_column] > 10:
            return (row[selected_date_column] / 12000) * 100
        elif row['Type'] == 'Team' and row[selected_date_column] > 10:
            return (row[selected_date_column] / 20000) * 100
        else:
            return None

    filtered_data['Target %'] = filtered_data.apply(calculate_target_percentage, axis=1)

    # Drop rows where Target % is null
    filtered_data = filtered_data.dropna(subset=['Target %'])

    # Define function to create a bar chart (Target %)
    def create_target_percentage_chart(data):
        fig = px.bar(
            data_frame=data,
            x="Terminal",  # Drill down from terminal
            y="Target %",  # Show calculated target %
            color="Type",  # Color by Type (Single/Team)
            hover_data=["Route", "UNIT NUMBER"],  # Add drill-down info
            barmode="group",
            title=f"Target Percentage by Terminal ({selected_date_column})"
        )
        fig.update_layout(xaxis_title="Terminal", yaxis_title="Target %")
        return fig

    # Plot the chart if data is available
    if not filtered_data.empty:
        st.write("### Filtered Trailer Data")
        st.write(filtered_data)
        
        st.write("### Target Percentage Visualization")
        fig = create_target_percentage_chart(filtered_data)
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected filters.")
else:
    st.warning("No data available for the selected filters.")

# Add an option to download filtered data as CSV
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_data.empty:
    csv = convert_df(filtered_data)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_mileage_data.csv', mime='text/csv')
