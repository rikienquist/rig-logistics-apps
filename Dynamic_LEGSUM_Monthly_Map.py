### driver pay works but not TLORDER

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_FREIGHT, LS_TRIP_NUMBER, LS_TO_ZONE, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

Use the following query to generate the required TLORDER data:  
SELECT BILL_NUMBER, CALLNAME, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE  
FROM TLORDER WHERE "PICK_UP_BY" BETWEEN 'X' AND 'Y';   

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file (optional)", type="csv")

@st.cache_data
def load_city_coordinates():
    city_coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    city_coords.rename(columns={
        "location": "LOCATION",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)
    return city_coords

@st.cache_data
def preprocess_legsum(file, coordinates):
    legsum_df = pd.read_csv(file)
    legsum_df = legsum_df.rename(columns={
        "LEGO_ZONE_DESC": "ORIG_LOCATION",
        "LEGD_ZONE_DESC": "DEST_LOCATION"
    })
    
    # Merge coordinates for origins and destinations
    origin_coords = coordinates.rename(columns={"LOCATION": "ORIG_LOCATION", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    legsum_df = legsum_df.merge(origin_coords, on="ORIG_LOCATION", how="left")

    dest_coords = coordinates.rename(columns={"LOCATION": "DEST_LOCATION", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    legsum_df = legsum_df.merge(dest_coords, on="DEST_LOCATION", how="left")

    return legsum_df

@st.cache_data
def preprocess_tlorder(file):
    return pd.read_csv(file)

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file and uploaded_tlorder_file:
    coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, coordinates_df)
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file)
    
    # Merge LEGSUM and TLORDER
    legsum_df = legsum_df.merge(
        tlorder_df[['BILL_NUMBER', 'CHARGES', 'XCHARGES', 'CURRENCY_CODE']],
        left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left'
    )
    legsum_df['CHARGES'] = pd.to_numeric(legsum_df['CHARGES'], errors='coerce')
    legsum_df['XCHARGES'] = pd.to_numeric(legsum_df['XCHARGES'], errors='coerce')
    exchange_rate = 1.38
    legsum_df['Total Charge (CAD)'] = np.where(
        legsum_df['CURRENCY_CODE'] == 'USD',
        (legsum_df['CHARGES'] + legsum_df['XCHARGES']) * exchange_rate,
        legsum_df['CHARGES'] + legsum_df['XCHARGES']
    )

    legsum_df['Straight Distance'] = calculate_haversine(legsum_df)

    # Convert LS_ACTUAL_DATE to datetime for filtering
    legsum_df['LS_ACTUAL_DATE'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE'], errors='coerce')

    # Get available date range
    available_dates = legsum_df['LS_ACTUAL_DATE'].dropna()
    min_date, max_date = available_dates.min(), available_dates.max()

    # Let user select Start and End Date within the relevant range
    start_date = st.date_input("Select Start Date:", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Select End Date:", value=max_date, min_value=min_date, max_value=max_date)

    # Filter data based on the selected date range
    filtered_df = legsum_df[
        (legsum_df['LS_ACTUAL_DATE'] >= pd.Timestamp(start_date)) &
        (legsum_df['LS_ACTUAL_DATE'] <= pd.Timestamp(end_date))
    ].copy()

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
    else:
        # Create route summary table
        filtered_df['Profit (CAD)'] = filtered_df['Total Charge (CAD)'] - filtered_df['LS_MT_LOADED']
        filtered_df['Revenue per Mile'] = filtered_df['Total Charge (CAD)'] / filtered_df['LS_LEG_DIST']
        route_summary = filtered_df[[
            'ORIG_LOCATION', 'DEST_LOCATION', 'LS_FREIGHT', 'LS_TRIP_NUMBER',
            'LS_LEG_DIST', 'LS_MT_LOADED', 'Total Charge (CAD)', 'Straight Distance',
            'Revenue per Mile', 'Profit (CAD)', 'LS_ACTUAL_DATE', 'LS_LEG_NOTE'
        ]].rename(columns={
            'ORIG_LOCATION': 'Origin',
            'DEST_LOCATION': 'Destination',
            'LS_FREIGHT': 'BILL_NUMBER',
            'LS_TRIP_NUMBER': 'Trip Number',
            'LS_LEG_DIST': 'Distance (miles)',
            'LS_MT_LOADED': 'Driver Pay (CAD)',
            'Straight Distance': 'Straight Distance (miles)'
        })

        # Display route summary
        st.dataframe(route_summary, use_container_width=True)

        # Generate map
        fig = go.Figure()
        for _, row in filtered_df.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
                mode="lines+markers",
                line=dict(width=2, color="blue"),
                marker=dict(size=8),
                hovertext=(
                    f"Origin: {row['ORIG_LOCATION']}<br>"
                    f"Destination: {row['DEST_LOCATION']}<br>"
                    f"Trip Number: {row['LS_TRIP_NUMBER']}<br>"
                    f"Distance: {row['LS_LEG_DIST']} miles<br>"
                    f"Straight Distance: {row['Straight Distance']:.2f} miles<br>"
                    f"Total Charge: ${row['Total Charge (CAD)']:.2f}<br>"
                    f"Driver Pay: ${row['LS_MT_LOADED']:.2f}<br>"
                    f"Profit: ${row['Profit (CAD)']:.2f}<br>"
                    f"Actual Date: {row['LS_ACTUAL_DATE'].strftime('%Y-%m-%d')}<br>"
                    f"Note: {row['LS_LEG_NOTE']}"
                ),
                hoverinfo="text"
            ))

        fig.update_layout(
            title="Truck Movements Map",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)

        # Identify missing locations
        missing_origins = filtered_df[pd.isna(filtered_df['ORIG_LAT'])][['ORIG_LOCATION']].drop_duplicates()
        missing_destinations = filtered_df[pd.isna(filtered_df['DEST_LAT'])][['DEST_LOCATION']].drop_duplicates()
        missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()
        if not missing_locations.empty:
            st.write("### Missing Locations")
            st.dataframe(missing_locations, use_container_width=True)
    else:
        st.warning("No data available for the selected date range.")
else:
    if not uploaded_legsum_file:
        st.warning("Please upload the LEGSUM CSV file.")
    if not uploaded_tlorder_file:
        st.warning("Please upload the TLORDER CSV file.")













