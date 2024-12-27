import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re
from datetime import datetime, timedelta

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
uploaded_tlorder_driverpay_file = st.file_uploader("Upload TLORDER+DRIVERPAY CSV file", type="csv", key="tlorder_driverpay")

@st.cache_data
def load_location_coordinates():
    location_coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    location_coords.rename(columns={
        "location": "ZONE_DESC",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)
    return location_coords

@st.cache_data
def preprocess_legsum(file, location_coords):
    df = pd.read_csv(file, low_memory=False)
    
    # Clean and standardize zone descriptions
    df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].str.strip().str.upper()
    df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].str.strip().str.upper()
    location_coords['ZONE_DESC'] = location_coords['ZONE_DESC'].str.strip().str.upper()

    # Merge for origin locations
    origin_coords = location_coords.copy()
    df = df.merge(origin_coords, left_on="LEGO_ZONE_DESC", right_on="ZONE_DESC", how="left")
    df = df.rename(columns={"LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    
    # Merge for destination locations
    dest_coords = location_coords.copy()
    df = df.merge(dest_coords, left_on="LEGD_ZONE_DESC", right_on="ZONE_DESC", how="left", suffixes=('', '_dest'))
    df = df.rename(columns={"LAT": "DEST_LAT", "LON": "DEST_LON"})
    
    # Convert date column
    df['LS_ACTUAL_DATE'] = pd.to_datetime(df['LS_ACTUAL_DATE'])
    
    return df

@st.cache_data
def preprocess_tlorder_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    # Convert numeric columns
    df['CHARGES'] = pd.to_numeric(df['CHARGES'], errors='coerce')
    df['XCHARGES'] = pd.to_numeric(df['XCHARGES'], errors='coerce')
    df['DISTANCE'] = pd.to_numeric(df['DISTANCE'], errors='coerce')
    df['TOTAL_PAY_SUM'] = pd.to_numeric(df['TOTAL_PAY_SUM'], errors='coerce')
    return df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Date selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", datetime.now())

if uploaded_legsum_file is not None and uploaded_tlorder_driverpay_file is not None:
    # Load and preprocess data
    location_coordinates = load_location_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, location_coordinates)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)
    
    # Filter by date range
    mask = (legsum_df['LS_ACTUAL_DATE'].dt.date >= start_date) & (legsum_df['LS_ACTUAL_DATE'].dt.date <= end_date)
    filtered_legsum = legsum_df[mask].copy()
    
    # Left join with TLORDER+DRIVERPAY data
    merged_df = filtered_legsum.merge(
        tlorder_driverpay_df,
        left_on='LS_FREIGHT',
        right_on='BILL_NUMBER',
        how='left'
    )

    # Calculate exchange rate adjusted charges
    exchange_rate = 1.38
    merged_df['TOTAL_CHARGE_CAD'] = np.where(
        merged_df['CURRENCY_CODE'] == 'USD',
        (merged_df['CHARGES'] + merged_df['XCHARGES']) * exchange_rate,
        merged_df['CHARGES'] + merged_df['XCHARGES']
    )
    
    # Calculate straight-line distance
    merged_df['Straight Distance'] = calculate_haversine(merged_df)
    
    # Get unique power units for selection
    power_unit_options = sorted(merged_df['LS_POWER_UNIT'].unique())
    selected_power_unit = st.selectbox("Select Power Unit:", options=power_unit_options)
    
    # Filter for selected power unit
    power_unit_data = merged_df[merged_df['LS_POWER_UNIT'] == selected_power_unit].copy()
    
    # Get relevant drivers for the selected power unit
    relevant_drivers = power_unit_data['LS_DRIVER'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver (optional):", options=driver_options)
    
    # Apply driver filter if specific driver selected
    if selected_driver != "All":
        power_unit_data = power_unit_data[power_unit_data['LS_DRIVER'] == selected_driver]

    if not power_unit_data.empty:
        # Create route summary DataFrame
        power_unit_data['Route'] = power_unit_data['LEGO_ZONE_DESC'] + " to " + power_unit_data['LEGD_ZONE_DESC']
        power_unit_data['Revenue per Mile'] = power_unit_data['TOTAL_CHARGE_CAD'] / power_unit_data['LS_LEG_DIST']
        power_unit_data['Profit (CAD)'] = power_unit_data['TOTAL_CHARGE_CAD'] - power_unit_data['TOTAL_PAY_SUM']

        route_summary_df = power_unit_data[[
            'Route', 'BILL_NUMBER', 'LS_TRIP_NUMBER', 'LS_LEG_DIST', 'LS_MT_LOADED',
            'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'Straight Distance', 'Revenue per Mile',
            'LS_ACTUAL_DATE', 'LS_LEG_NOTE'
        ]].copy()

        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            'Route': 'Grand Totals',
            'BILL_NUMBER': '',
            'LS_TRIP_NUMBER': '',
            'LS_LEG_DIST': route_summary_df['LS_LEG_DIST'].sum(),
            'LS_MT_LOADED': '',
            'TOTAL_CHARGE_CAD': route_summary_df['TOTAL_CHARGE_CAD'].sum(),
            'Straight Distance': route_summary_df['Straight Distance'].sum(),
            'Revenue per Mile': route_summary_df['TOTAL_CHARGE_CAD'].sum() / route_summary_df['LS_LEG_DIST'].sum()
            if route_summary_df['LS_LEG_DIST'].sum() != 0 else 0,
            'LS_ACTUAL_DATE': None,
            'LS_LEG_NOTE': ''
        }])

        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)

        # Format currency and numeric columns
        for col in ['TOTAL_CHARGE_CAD', 'Revenue per Mile']:
            route_summary_df[col] = route_summary_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )

        # Display route summary
        st.write("Route Summary:")
        st.dataframe(route_summary_df, use_container_width=True)

        # Create the map
        fig = go.Figure()
        
        # Track already added locations for legend
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        # Create location aggregates for hover information
        location_aggregates = {}
        for _, row in power_unit_data.iterrows():
            # Aggregate origin data
            if row['LEGO_ZONE_DESC'] not in location_aggregates:
                location_aggregates[row['LEGO_ZONE_DESC']] = {
                    'total_charge': 0,
                    'total_distance': 0,
                    'total_pay': 0,
                    'total_profit': 0,
                    'count': 0
                }
            
            # Aggregate destination data
            if row['LEGD_ZONE_DESC'] not in location_aggregates:
                location_aggregates[row['LEGD_ZONE_DESC']] = {
                    'total_charge': 0,
                    'total_distance': 0,
                    'total_pay': 0,
                    'total_profit': 0,
                    'count': 0
                }
            
            # Update aggregates for both origin and destination
            for location in [row['LEGO_ZONE_DESC'], row['LEGD_ZONE_DESC']]:
                location_aggregates[location]['total_charge'] += row['TOTAL_CHARGE_CAD'] if pd.notna(row['TOTAL_CHARGE_CAD']) else 0
                location_aggregates[location]['total_distance'] += row['LS_LEG_DIST'] if pd.notna(row['LS_LEG_DIST']) else 0
                location_aggregates[location]['total_pay'] += row['TOTAL_PAY_SUM'] if pd.notna(row['TOTAL_PAY_SUM']) else 0
                location_aggregates[location]['total_profit'] += row['Profit (CAD)'] if pd.notna(row['Profit (CAD)']) else 0
                location_aggregates[location]['count'] += 1

        # Add traces to the map
        for _, row in power_unit_data.iterrows():
            # Create hover text for origin
            origin_agg = location_aggregates[row['LEGO_ZONE_DESC']]
            origin_hover = (
                f"Location: {row['LEGO_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${origin_agg['total_charge']:,.2f}<br>"
                f"Total Distance: {origin_agg['total_distance']:,.1f} miles<br>"
                f"Revenue per Mile: ${(origin_agg['total_charge']/origin_agg['total_distance']):,.2f}<br>"
                f"Total Driver Pay: ${origin_agg['total_pay']:,.2f}<br>"
                f"Total Profit: ${origin_agg['total_profit']:,.2f}"
            )

            # Add origin marker
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON']],
                lat=[row['ORIG_LAT']],
                mode="markers",
                marker=dict(size=8, color="blue"),
                name="Origin" if not legend_added["Origin"] else None,
                hoverinfo="text",
                hovertext=origin_hover,
                showlegend=not legend_added["Origin"]
            ))
            legend_added["Origin"] = True

            # Create hover text for destination
            dest_agg = location_aggregates[row['LEGD_ZONE_DESC']]
            dest_hover = (
                f"Location: {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${dest_agg['total_charge']:,.2f}<br>"
                f"Total Distance: {dest_agg['total_distance']:,.1f} miles<br>"
                f"Revenue per Mile: ${(dest_agg['total_charge']/dest_agg['total_distance']):,.2f}<br>"
                f"Total Driver Pay: ${dest_agg['total_pay']:,.2f}<br>"
                f"Total Profit: ${dest_agg['total_profit']:,.2f}"
            )

            # Add destination marker
            fig.add_trace(go.Scattergeo(
                lon=[row['DEST_LON']],
                lat=[row['DEST_LAT']],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Destination" if not legend_added["Destination"] else None,
                hoverinfo="text",
                hovertext=dest_hover,
                showlegend=not legend_added["Destination"]
            ))
            legend_added["Destination"] = True

            # Add route line
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route" if not legend_added["Route"] else None,
                hoverinfo="skip",
                showlegend=not legend_added["Route"]
            ))
            legend_added["Route"] = True

        # Update map layout
        fig.update_layout(
            title=f"Routes for Power Unit: {selected_power_unit}, Driver: {selected_driver}",
            geo=dict(
                scope="north america",
                projection_type="mercator",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)"
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

        # Display locations missing coordinates
        missing_origins = power_unit_data[
            pd.isna(power_unit_data['ORIG_LAT']) | pd.isna(power_unit_data['ORIG_LON'])
        ]['LEGO_ZONE_DESC'].unique()

        missing_destinations = power_unit_data[
            pd.isna(power_unit_data['DEST_LAT']) | pd.isna(power_unit_data['DEST_LON'])
        ]['LEGD_ZONE_DESC'].unique()

        missing_locations = pd.DataFrame(
            sorted(set(missing_origins) | set(missing_destinations)),
            columns=['Location']
        )

        if not missing_locations.empty:
            st.write("### Missing Locations")
            st.write("The following locations are missing coordinates in the location_coordinates.csv file:")
            st.dataframe(missing_locations, use_container_width=True)

    else:
        st.warning("No data available for the selected Power Unit and Driver.")

else:
    st.warning("Please upload both the LEGSUM and TLORDER+DRIVERPAY CSV files to proceed.")




            
