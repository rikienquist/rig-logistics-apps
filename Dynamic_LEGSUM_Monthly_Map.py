import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Streamlit App Title and Instructions
st.title("Trip Map Viewer")

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
uploaded_tlorder_driverpay_file = st.file_uploader("Upload TLORDER + DRIVERPAY CSV file", type="csv")

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
    legsum_df = pd.read_csv(file, low_memory=False)
    legsum_df['LS_ACTUAL_DATE'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE'])
    
    # Merge LEGO_ZONE_DESC and LEGD_ZONE_DESC with location coordinates
    lego_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC"})
    legsum_df = legsum_df.merge(lego_coords, on="LEGO_ZONE_DESC", how="left").rename(
        columns={"LAT": "ORIG_LAT", "LON": "ORIG_LON"}
    )

    legd_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC"})
    legsum_df = legsum_df.merge(legd_coords, on="LEGD_ZONE_DESC", how="left").rename(
        columns={"LAT": "DEST_LAT", "LON": "DEST_LON"}
    )

    return legsum_df

@st.cache_data
def preprocess_tlorder_driverpay(file):
    tlorder_driverpay_df = pd.read_csv(file, low_memory=False)
    tlorder_driverpay_df['TOTAL_PAY_SUM'] = pd.to_numeric(tlorder_driverpay_df['TOTAL_PAY_SUM'], errors='coerce')
    tlorder_driverpay_df['CHARGES'] = pd.to_numeric(tlorder_driverpay_df['CHARGES'], errors='coerce')
    tlorder_driverpay_df['XCHARGES'] = pd.to_numeric(tlorder_driverpay_df['XCHARGES'], errors='coerce')
    tlorder_driverpay_df['TOTAL_CHARGE_CAD'] = tlorder_driverpay_df['CHARGES'] + tlorder_driverpay_df['XCHARGES']
    return tlorder_driverpay_df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Determine date range from LS_ACTUAL_DATE in LEGSUM
    min_date = legsum_df['LS_ACTUAL_DATE'].min()
    max_date = legsum_df['LS_ACTUAL_DATE'].max()

    # Allow user to select date range within the available data
    start_date = st.date_input("Start Date:", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date:", value=max_date, min_value=min_date, max_value=max_date)

    filtered_df = legsum_df[
        (legsum_df['LS_ACTUAL_DATE'] >= pd.to_datetime(start_date)) &
        (legsum_df['LS_ACTUAL_DATE'] <= pd.to_datetime(end_date))
    ]

    # Merge LEGSUM and TLORDER+DRIVERPAY data on LS_FREIGHT and BILL_NUMBER
    merged_df = filtered_df.merge(
        tlorder_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left'
    )

    merged_df['Straight Distance'] = calculate_haversine(merged_df)

    # Calculate Revenue per Mile and Profit
    merged_df['Revenue per Mile'] = merged_df['TOTAL_CHARGE_CAD'] / merged_df['LS_LEG_DIST']
    merged_df['Profit (CAD)'] = merged_df['TOTAL_CHARGE_CAD'] - merged_df['TOTAL_PAY_SUM']

    # Prepare the route summary
    route_summary_df = merged_df.assign(
        Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
    )[
        [
            "Route", "BILL_NUMBER", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED",
            "TOTAL_CHARGE_CAD", "LS_ACTUAL_DATE", "Straight Distance", "Revenue per Mile", "Profit (CAD)", "LS_LEG_NOTE"
        ]
    ]
    
    route_summary_df.rename(columns={
        "TOTAL_CHARGE_CAD": "Total Charge (CAD)", 
        "LS_LEG_DIST": "Distance (miles)"
    }, inplace=True)

    # Display route summary
    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)

        # Check for missing coordinates
    missing_origins = merged_df[
        pd.isna(merged_df['ORIG_LAT']) | pd.isna(merged_df['ORIG_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Missing Location'})

    missing_destinations = merged_df[
        pd.isna(merged_df['DEST_LAT']) | pd.isna(merged_df['DEST_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Missing Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Generate the map
    fig = go.Figure()

    for _, row in merged_df.iterrows():
        # Add origin marker
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON']],
            lat=[row['ORIG_LAT']],
            mode="markers",
            marker=dict(size=8, color="blue"),
            name="Origin",
            hoverinfo="text",
            hovertext=(
                f"City: {row['LEGO_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:,.2f}<br>"
                f"Distance (miles): {row['LS_LEG_DIST']:,.1f}<br>"
                f"Revenue per Mile: ${row['Revenue per Mile']:,.2f}<br>"
                f"Driver Pay (CAD): ${row['TOTAL_PAY_SUM']:,.2f}<br>"
                f"Profit (CAD): ${row['Profit (CAD)']:,.2f}"
            )
        ))

        # Add destination marker
        fig.add_trace(go.Scattergeo(
            lon=[row['DEST_LON']],
            lat=[row['DEST_LAT']],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Destination",
            hoverinfo="text",
            hovertext=(
                f"City: {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:,.2f}<br>"
                f"Distance (miles): {row['LS_LEG_DIST']:,.1f}<br>"
                f"Revenue per Mile: ${row['Revenue per Mile']:,.2f}<br>"
                f"Driver Pay (CAD): ${row['TOTAL_PAY_SUM']:,.2f}<br>"
                f"Profit (CAD): ${row['Profit (CAD)']:,.2f}"
            )
        ))

        # Add route line
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Route",
            hoverinfo="skip"
        ))

    # Configure the map layout
    fig.update_layout(
        title="Trip Routes",
        geo=dict(
            scope="north america",
            projection_type="mercator",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)"
        ),
    )

    # Display the map
    st.plotly_chart(fig)

    # Show missing locations
    if not missing_locations.empty:
        st.write("### Missing Locations")
        st.dataframe(missing_locations, use_container_width=True)

else:
    st.warning("Please upload both the LEGSUM and TLORDER+DRIVERPAY CSV files to proceed.")
