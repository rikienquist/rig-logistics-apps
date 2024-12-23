import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Streamlit App Title and Instructions
st.title("Trip Map Viewer - LEGSUM")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_FREIGHT, LS_TRIP_NUMBER, LS_TO_ZONE, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

Use the following query to generate the required TLORDER data:  
SELECT BILL_NUMBER, CALLNAME, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE  
FROM TLORDER WHERE "PICK_UP_BY" BETWEEN 'X' AND 'Y';  

Use the following query to generate the required DRIVERPAY data:  
SELECT BILL_NUMBER, PAY_ID, DRIVER_ID, TOTAL_PAY_AMT, PAID_DATE  
FROM DRIVERPAY WHERE "PAID_DATE" BETWEEN 'X' AND 'Y';  

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file", type="csv")
uploaded_driverpay_file = st.file_uploader("Upload DRIVERPAY CSV file", type="csv")

@st.cache_data
def load_city_coordinates():
    location_coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    location_coords.rename(columns={"location": "LOCATION", "latitude": "LAT", "longitude": "LON"}, inplace=True)
    return location_coords

@st.cache_data
def preprocess_data(legsum_file, tlorder_file, driverpay_file, location_coords):
    legsum = pd.read_csv(legsum_file)
    tlorder = pd.read_csv(tlorder_file).rename(columns={"BILL_NUMBER": "TLORDER_BILL_NUMBER"})
    driverpay = pd.read_csv(driverpay_file).rename(columns={"BILL_NUMBER": "DRIVERPAY_BILL_NUMBER"})
    
    # Merge LEGSUM with TLORDER and DRIVERPAY
    data = legsum.merge(tlorder, left_on="LS_FREIGHT", right_on="TLORDER_BILL_NUMBER", how="left")
    data = data.merge(driverpay, left_on="LS_FREIGHT", right_on="DRIVERPAY_BILL_NUMBER", how="left")
    
    # Clean and map coordinates
    location_coords["LOCATION"] = location_coords["LOCATION"].str.upper().str.strip()
    data["LEGO_ZONE_DESC"] = data["LEGO_ZONE_DESC"].str.upper().str.strip()
    data["LEGD_ZONE_DESC"] = data["LEGD_ZONE_DESC"].str.upper().str.strip()
    
    origin_coords = location_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    dest_coords = location_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    
    data = data.merge(origin_coords, on="LEGO_ZONE_DESC", how="left")
    data = data.merge(dest_coords, on="LEGD_ZONE_DESC", how="left")
    
    return data

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df["ORIG_LAT"]), np.radians(df["ORIG_LON"])
    lat2, lon2 = np.radians(df["DEST_LAT"]), np.radians(df["DEST_LON"])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file and uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    
    # Preprocess LEGSUM
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    
    # Preprocess and merge TLORDER
    tlorder_df = pd.read_csv(uploaded_tlorder_file).rename(columns={"BILL_NUMBER": "TLORDER_BILL_NUMBER"})
    legsum_df = legsum_df.merge(tlorder_df, left_on='LS_FREIGHT', right_on='TLORDER_BILL_NUMBER', how='left')
    
    # Preprocess and merge DRIVERPAY
    driverpay_df = pd.read_csv(uploaded_driverpay_file).rename(columns={"BILL_NUMBER": "DRIVERPAY_BILL_NUMBER"})
    legsum_df = legsum_df.merge(driverpay_df, left_on='LS_FREIGHT', right_on='DRIVERPAY_BILL_NUMBER', how='left')

    # Add currency conversion for charges (if applicable)
    exchange_rate = 1.38
    legsum_df['CHARGES'] = pd.to_numeric(legsum_df.get('CHARGES', None), errors='coerce')
    legsum_df['XCHARGES'] = pd.to_numeric(legsum_df.get('XCHARGES', None), errors='coerce')

    # Calculate TOTAL_CHARGE_CAD for rows with valid TLORDER_BILL_NUMBER
    legsum_df['TOTAL_CHARGE_CAD'] = np.where(
        pd.notna(legsum_df['TLORDER_BILL_NUMBER']),
        (legsum_df['CHARGES'].fillna(0) + legsum_df['XCHARGES'].fillna(0)) * exchange_rate,
        None
    )

    # Handle LS_LEG_DIST and calculate Revenue per Mile
    legsum_df['LS_LEG_DIST'] = pd.to_numeric(legsum_df['LS_LEG_DIST'], errors='coerce')
    legsum_df['LS_LEG_DIST'] = np.where(legsum_df['LS_LEG_DIST'] > 0, legsum_df['LS_LEG_DIST'], np.nan)

    legsum_df['Revenue per Mile'] = np.where(
        pd.notna(legsum_df['TOTAL_CHARGE_CAD']) & pd.notna(legsum_df['LS_LEG_DIST']),
        legsum_df['TOTAL_CHARGE_CAD'] / legsum_df['LS_LEG_DIST'],
        np.nan
    )

    # Calculate Profit (CAD)
    legsum_df['Profit (CAD)'] = np.where(
        pd.notna(legsum_df['TOTAL_CHARGE_CAD']),
        legsum_df['TOTAL_CHARGE_CAD'] - legsum_df['TOTAL_PAY_AMT'].fillna(0),
        np.nan
    )

    # Filter by date
    legsum_df['LS_ACTUAL_DATE'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE'], errors='coerce')
    start_date = st.date_input("Start Date", value=legsum_df['LS_ACTUAL_DATE'].min())
    end_date = st.date_input("End Date", value=legsum_df['LS_ACTUAL_DATE'].max())
    filtered_legsum = legsum_df[
        (legsum_df['LS_ACTUAL_DATE'] >= start_date) & (legsum_df['LS_ACTUAL_DATE'] <= end_date)
    ]

    # Identify missing locations
    missing_origins = filtered_legsum[
        pd.isna(filtered_legsum['LEGO_LAT']) | pd.isna(filtered_legsum['LEGO_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = filtered_legsum[
        pd.isna(filtered_legsum['LEGD_LAT']) | pd.isna(filtered_legsum['LEGD_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Calculate Straight Distance
    filtered_legsum['Straight Distance'] = np.where(
        pd.isna(filtered_legsum['LEGO_LAT']) | pd.isna(filtered_legsum['LEGD_LAT']),
        np.nan,
        calculate_haversine(filtered_legsum)
    )

    # Select power unit and driver
    filtered_legsum['LS_POWER_UNIT'] = filtered_legsum['LS_POWER_UNIT'].astype(str)
    punit_options = sorted(filtered_legsum['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)

    relevant_drivers = filtered_legsum[filtered_legsum['LS_POWER_UNIT'] == selected_punit]['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID:", options=driver_options)

    filtered_view = filtered_legsum[filtered_legsum['LS_POWER_UNIT'] == selected_punit]
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver]

    # Route summary
    if not filtered_view.empty:
        filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_AMT']
        route_summary = filtered_view.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[
            ['Route', 'LS_FREIGHT', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'Straight Distance',
             'Revenue per Mile', 'DRIVER_ID', 'TOTAL_PAY_AMT', 'Profit (CAD)', 'LS_ACTUAL_DATE', 'LS_LEG_NOTE']
        ].rename(columns={
            'LS_FREIGHT': 'BILL_NUMBER',
            'TOTAL_CHARGE_CAD': 'Total Charge (CAD)',
            'LS_LEG_DIST': 'Distance (miles)',
            'Straight Distance': 'Straight Distance (miles)',
            'TOTAL_PAY_AMT': 'Driver Pay (CAD)'
        })

        st.dataframe(route_summary)

    # Missing locations
    if not missing_locations.empty:
        st.write("### Locations Missing Coordinates")
        st.dataframe(missing_locations)

else:
    st.warning("Please upload LEGSUM, TLORDER, and DRIVERPAY CSV files to proceed.")
