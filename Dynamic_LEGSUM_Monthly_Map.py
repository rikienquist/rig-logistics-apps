import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "date_range" not in st.session_state:
    st.session_state.date_range = None

# Streamlit App Title and Instructions
st.title("Trip Map Viewer by Date Range")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_DRIVER, LS_FREIGHT, LS_TRIP_NUMBER, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

Use the following query to generate the required TLORDER + DRIVERPAY data:  
SELECT 
    D.BILL_NUMBER, O.CALLNAME, O.CHARGES, O.XCHARGES, O.DISTANCE,
    O.DISTANCE_UNITS, O.CURRENCY_CODE, SUM(D.TOTAL_PAY_AMT) AS TOTAL_PAY_SUM
FROM 
    TLORDER O
RIGHT JOIN 
    DRIVERPAY D
ON 
    O.BILL_NUMBER = D.BILL_NUMBER
WHERE 
    O.PICK_UP_BY BETWEEN 'X' AND 'Y'
GROUP BY 
    D.BILL_NUMBER, O.CALLNAME, O.CHARGES, O.XCHARGES, O.DISTANCE, O.DISTANCE_UNITS, O.CURRENCY_CODE;

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")

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
def preprocess_legsum(file, city_coords):
    df = pd.read_csv(file, low_memory=False)
    
    # Clean and standardize location names
    def clean_location(location):
        return re.sub(r"[^a-zA-Z\s]", "", str(location)).strip().upper()
    
    city_coords['LOCATION'] = city_coords['LOCATION'].apply(clean_location)
    df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].apply(clean_location)
    df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].apply(clean_location)

    # Merge with city coordinates for origins and destinations
    origin_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(origin_coords, on="LEGO_ZONE_DESC", how="left")
    
    dest_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    df = df.merge(dest_coords, on="LEGD_ZONE_DESC", how="left")
    
    return df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)

    # Date range selection
    start_date, end_date = st.date_input("Select Date Range:", [pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31")])
    filtered_df = legsum_df[
        (pd.to_datetime(legsum_df['LS_ACTUAL_DATE']) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(legsum_df['LS_ACTUAL_DATE']) <= pd.to_datetime(end_date))
    ]

    # Route summary table
    route_summary_df = filtered_df.assign(
        Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
    )[
        ['Route', 'LS_FREIGHT', 'LS_TRIP_NUMBER', 'LS_LEG_DIST', 'LS_MT_LOADED', 'LS_ACTUAL_DATE', 'LS_LEG_NOTE']
    ]

    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)

    # Map visualization
    fig = go.Figure()
    for _, row in filtered_df.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode="markers+lines",
            marker=dict(size=8),
            name="Route",
            hovertext=f"Route: {row['LEGO_ZONE_DESC']} to {row['LEGD_ZONE_DESC']}<br>"
                      f"Freight: {row['LS_FREIGHT']}<br>"
                      f"Trip Number: {row['LS_TRIP_NUMBER']}<br>"
                      f"Distance: {row['LS_LEG_DIST']} miles<br>"
                      f"MT Loaded: {row['LS_MT_LOADED']}<br>"
                      f"Note: {row['LS_LEG_NOTE']}<br>"
                      f"Date: {row['LS_ACTUAL_DATE']}"
        ))

    fig.update_layout(
        title="Truck Movements",
        geo=dict(scope="north america", projection_type="mercator"),
    )
    st.plotly_chart(fig)

    # Missing Locations
    missing_origins = filtered_df[
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['ORIG_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Missing Location'})

    missing_destinations = filtered_df[        pd.isna(filtered_df['DEST_LAT']) | pd.isna(filtered_df['DEST_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Missing Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    if not missing_locations.empty:
        st.write("### Missing Locations")
        st.dataframe(missing_locations, use_container_width=True)
else:
    st.warning("Please upload the LEGSUM CSV file to proceed.")











