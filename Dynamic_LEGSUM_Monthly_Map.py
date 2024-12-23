import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Streamlit App Title and Instructions
st.title("Truck Route Map Viewer")

st.markdown("""
### Instructions:
1. Use the following queries to extract the required data:
   - **LEGSUM**:
     ```sql
     SELECT LS_POWER_UNIT, LS_FREIGHT, LS_TRIP_NUMBER, LS_TO_ZONE, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
            LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
     FROM LEGSUM WHERE LS_ACTUAL_DATE BETWEEN 'X' AND 'Y';
     ```
   - **TLORDER**:
     ```sql
     SELECT BILL_NUMBER, CALLNAME, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE  
     FROM TLORDER WHERE PICK_UP_BY BETWEEN 'X' AND 'Y';
     ```
   - **DRIVERPAY**:
     ```sql
     SELECT BILL_NUMBER, PAY_ID, DRIVER_ID, TOTAL_PAY_AMT, PAID_DATE  
     FROM DRIVERPAY WHERE PAID_DATE BETWEEN 'X' AND 'Y';
     ```
2. Replace `X` and `Y` with the desired date range in `YYYY-MM-DD` format.
3. Save the query results as CSV files and upload them below.

**Note**: Coordinates are matched from the `location_coordinates.csv` file.
""")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file", type="csv")
uploaded_driverpay_file = st.file_uploader("Upload DRIVERPAY CSV file", type="csv")

@st.cache_data
def load_city_coordinates():
    coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    coords.rename(columns={
        "location": "LOCATION",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)
    return coords

@st.cache_data
def preprocess_legsum(file, tlorder, driverpay, city_coords):
    legsum = pd.read_csv(file)
    tlorder = pd.read_csv(tlorder)
    driverpay = pd.read_csv(driverpay)

    # Rename columns in TLORDER and DRIVERPAY to prevent merging conflicts
    tlorder.rename(columns={"BILL_NUMBER": "TL_BILL_NUMBER"}, inplace=True)
    driverpay.rename(columns={"BILL_NUMBER": "DP_BILL_NUMBER"}, inplace=True)

    # Merge LEGSUM with TLORDER and DRIVERPAY
    merged = legsum.merge(tlorder, left_on="LS_FREIGHT", right_on="TL_BILL_NUMBER", how="left")
    merged = merged.merge(driverpay, left_on="LS_FREIGHT", right_on="DP_BILL_NUMBER", how="left")

    # Clean city names and merge with coordinates
    city_coords["LOCATION"] = city_coords["LOCATION"].str.upper()
    merged = merged.merge(city_coords, left_on="LEGO_ZONE_DESC", right_on="LOCATION", how="left", suffixes=('', '_ORIG'))
    merged = merged.merge(city_coords, left_on="LEGD_ZONE_DESC", right_on="LOCATION", how="left", suffixes=('_ORIG', '_DEST'))

    # Calculate Total Charge (CAD)
    exchange_rate = 1.38
    merged['CHARGES'] = pd.to_numeric(merged['CHARGES'], errors='coerce')
    merged['XCHARGES'] = pd.to_numeric(merged['XCHARGES'], errors='coerce')
    merged['TOTAL_CHARGE_CAD'] = np.where(
        merged['CURRENCY_CODE'] == 'USD',
        (merged['CHARGES'] + merged['XCHARGES']) * exchange_rate,
        merged['CHARGES'] + merged['XCHARGES']
    )

    # Calculate Profit
    merged['TOTAL_PAY_AMT'] = pd.to_numeric(merged['TOTAL_PAY_AMT'], errors='coerce')
    merged['Profit (CAD)'] = merged['TOTAL_CHARGE_CAD'] - merged['TOTAL_PAY_AMT']

    # Calculate Straight Distance
    R = 3958.8
    lat1, lon1 = np.radians(merged['LAT_ORIG']), np.radians(merged['LON_ORIG'])
    lat2, lon2 = np.radians(merged['LAT_DEST']), np.radians(merged['LON_DEST'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    merged['Straight Distance (miles)'] = R * c

    merged['Revenue per Mile'] = merged['TOTAL_CHARGE_CAD'] / merged['LS_LEG_DIST']

    return merged

if uploaded_legsum_file and uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates = load_city_coordinates()
    processed_data = preprocess_legsum(uploaded_legsum_file, uploaded_tlorder_file, uploaded_driverpay_file, city_coordinates)

    st.sidebar.header("Date Range")
    start_date = st.sidebar.date_input("Start Date", value=processed_data['LS_ACTUAL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=processed_data['LS_ACTUAL_DATE'].max())

    filtered_data = processed_data[
        (processed_data['LS_ACTUAL_DATE'] >= pd.to_datetime(start_date)) &
        (processed_data['LS_ACTUAL_DATE'] <= pd.to_datetime(end_date))
    ]

    if not filtered_data.empty:
        # Route Summary
        route_summary = filtered_data.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[[
            "Route", "LS_FREIGHT", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED",
            "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Straight Distance (miles)",
            "Revenue per Mile", "TOTAL_PAY_AMT", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE"
        ]].rename(columns={
            "LS_FREIGHT": "BILL_NUMBER",
            "LS_LEG_DIST": "Distance (miles)",
            "TOTAL_PAY_AMT": "Driver Pay (CAD)"
        })
        st.write("### Route Summary")
        st.dataframe(route_summary)

        # Missing Locations
        missing_locations = filtered_data[
            filtered_data[['LAT_ORIG', 'LON_ORIG', 'LAT_DEST', 'LON_DEST']].isnull().any(axis=1)
        ][['LEGO_ZONE_DESC', 'LEGD_ZONE_DESC']].drop_duplicates()

        if not missing_locations.empty:
            st.write("### Missing Locations")
            st.dataframe(missing_locations)

        # Generate Map
        fig = go.Figure()
        for _, row in filtered_data.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[row['LON_ORIG'], row['LON_DEST']],
                lat=[row['LAT_ORIG'], row['LAT_DEST']],
                mode="lines+markers",
                line=dict(width=2, color="blue"),
                marker=dict(size=5),
                hoverinfo="text",
                hovertext=f"""
                Route: {row['LEGO_ZONE_DESC']} to {row['LEGD_ZONE_DESC']}<br>
                Distance: {row['LS_LEG_DIST']} miles<br>
                Total Charge (CAD): {row['TOTAL_CHARGE_CAD']:.2f}<br>
                Profit (CAD): {row['Profit (CAD)']:.2f}
                """
            ))
        fig.update_layout(
            title="Truck Routes",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected date range.")
else:
    st.warning("Please upload LEGSUM, TLORDER, and DRIVERPAY CSV files to proceed.")
