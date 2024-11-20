import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from geopy.distance import geodesic
import os

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Load the city coordinates from the consolidated file
city_coordinates_file = "trip_map_data/city_coordinates.csv"
city_coordinates_df = pd.read_csv(city_coordinates_file)

# Load other datasets
data_folder = "trip_map_data"
tlorder_df = pd.read_csv(os.path.join(data_folder, "TLORDER_Sep2022-Sep2024_V3.csv"), low_memory=False)
driver_pay_df = pd.read_csv(os.path.join(data_folder, "driver_pay_data.csv"), low_memory=False)

# Function to get latitude and longitude from city and province
def get_coordinates(city, province):
    match = city_coordinates_df[
        (city_coordinates_df['city'] == city.strip().upper()) &
        (city_coordinates_df['province'] == province.strip().upper())
    ]
    if not match.empty:
        return match.iloc[0]['latitude'], match.iloc[0]['longitude']
    else:
        return None, None  # Return None if coordinates are not found

# Add coordinates to the TLORDER dataframe
def add_coordinates(df):
    # Get origin coordinates
    df['ORIG_LAT'], df['ORIG_LON'] = zip(*df.apply(
        lambda row: get_coordinates(row['ORIGCITY'], row['ORIGPROV']), axis=1
    ))
    # Get destination coordinates
    df['DEST_LAT'], df['DEST_LON'] = zip(*df.apply(
        lambda row: get_coordinates(row['DESTCITY'], row['DESTPROV']), axis=1
    ))

add_coordinates(tlorder_df)

# Filter for non-same-city routes
tlorder_df = tlorder_df[(tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) & 
                        (pd.notna(tlorder_df['ORIG_LAT'])) & 
                        (pd.notna(tlorder_df['DEST_LAT']))].copy()

# Merge with driver pay
driver_pay_agg = driver_pay_df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum', 'DRIVER_ID': 'first'})
tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

# Calculate CAD charge and filter
tlorder_df['TOTAL_CHARGE_CAD'] = tlorder_df.apply(
    lambda x: (x['CHARGES'] + x['XCHARGES']) * 1.38 if x['CURRENCY_CODE'] == 'USD' else x['CHARGES'] + x['XCHARGES'], 
    axis=1
)
filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & (tlorder_df['DISTANCE'] != 0)].copy()

# Ensure PICK_UP_PUNIT is clean
filtered_df['PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].astype(str).fillna("Unknown")

# Calculate Revenue per Mile and Profit Margin
filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df['Profit Margin (%)'] = (filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['TOTAL_PAY_AMT']) * 100

# Add Month Column for Grouping
filtered_df['PICK_UP_DATE'] = pd.to_datetime(filtered_df['PICK_UP_BY'])
filtered_df['Month'] = filtered_df['PICK_UP_DATE'].dt.to_period('M')

# Streamlit App
st.title("Trip Map Viewer by Month")

# PUNIT and Driver ID selection
punit_options = sorted(filtered_df['PICK_UP_PUNIT'].dropna().unique())
selected_punit = st.selectbox("Select PUNIT:", options=punit_options)

driver_options = ["All"] + sorted(filtered_df['DRIVER_ID'].dropna().astype(str))
selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)

# Filter based on selections
filtered_view = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit].copy()
if selected_driver != "All":
    filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()

# Month navigation
months = sorted(filtered_view['Month'].dropna().unique())
total_months = len(months)

def navigate_months(direction):
    if direction == "previous" and st.session_state.month_index > 0:
        st.session_state.month_index -= 1
    elif direction == "next" and st.session_state.month_index < total_months - 1:
        st.session_state.month_index += 1
    elif direction == "back_12" and st.session_state.month_index > 11:
        st.session_state.month_index -= 12
    elif direction == "ahead_12" and st.session_state.month_index < total_months - 12:
        st.session_state.month_index += 12

col1, col2, col3, col4 = st.columns(4)
col1.button("Previous Month", on_click=navigate_months, args=("previous",))
col2.button("Next Month", on_click=navigate_months, args=("next",))
col3.button("Back 12 Months", on_click=navigate_months, args=("back_12",))
col4.button("Ahead 12 Months", on_click=navigate_months, args=("ahead_12",))

if total_months > 0:
    selected_month = months[st.session_state.month_index]
    st.write(f"Viewing data for month: {selected_month}")
    month_data = filtered_view[filtered_view['Month'] == selected_month].copy()

    # Generate map
    fig = go.Figure()
    month_data = month_data.sort_values(by='PICK_UP_DATE')  # Sort routes chronologically
    label_counter = 1
    for i, row in month_data.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON']],
            lat=[row['ORIG_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="blue"),
            text=str(label_counter),
            textposition="top right",
            name="Origin" if label_counter == 1 else None,  # Legend only for the first point
            hovertext=(f"City: {row['ORIGCITY']}, {row['ORIGPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Distance (miles): {row['DISTANCE']}<br>"),
            hoverinfo="text",
        ))
        fig.add_trace(go.Scattergeo(
            lon=[row['DEST_LON']],
            lat=[row['DEST_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=str(label_counter + 1),
            textposition="top right",
            name="Destination" if label_counter == 1 else None,  # Legend only for the first point
            hovertext=(f"City: {row['DESTCITY']}, {row['DESTPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Distance (miles): {row['DISTANCE']}<br>"),
            hoverinfo="text",
        ))
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Route" if label_counter == 1 else None,  # Legend only for the first route
            hoverinfo="skip",
        ))
        label_counter += 2

    fig.update_layout(
        title=f"Routes for {selected_month} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
        geo=dict(scope="north america", projection_type="mercator"),
    )
    st.plotly_chart(fig)

    # Create the route summary table
    route_summary = []
    for _, row in month_data.iterrows():
        route_summary.append({
            "Route": f"{row['ORIGCITY']}, {row['ORIGPROV']} to {row['DESTCITY']}, {row['DESTPROV']}",
            "BILL_NUMBER": row['BILL_NUMBER'],
            "Total Charge (CAD)": f"${row['TOTAL_CHARGE_CAD']:.2f}",
            "Distance (miles)": row['DISTANCE'],
            "Revenue per Mile": f"${row['Revenue per Mile']:.2f}",
            "Driver ID": row['DRIVER_ID'],
            "Driver Pay (CAD)": f"${row['TOTAL_PAY_AMT']:.2f}" if not pd.isna(row['TOTAL_PAY_AMT']) else "N/A",
            "Profit Margin (%)": f"{row['Profit Margin (%)']:.2f}%" if not pd.isna(row['Profit Margin (%)']) else "N/A",
            "Geopy_Distance": row['Geopy_Distance'],
            "Date": row['PICK_UP_DATE']
        })

    # Convert the route summary to a DataFrame
    route_summary_df = pd.DataFrame(route_summary)

    # Display the table
    st.write("Route Summary:")
    st.dataframe(route_summary_df)
else:
    st.warning("No data available for the selected PUNIT and Driver ID.")
