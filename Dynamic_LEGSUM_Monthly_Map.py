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
    
    # Clean location names
    def clean_location_name(name):
        return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().upper()

    # Clean and standardize location names
    city_coords['LOCATION'] = city_coords['LOCATION'].apply(clean_location_name)
    df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].apply(clean_location_name)
    df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].apply(clean_location_name)

    # Ensure there are no duplicates in city_coords after cleaning
    city_coords = city_coords.drop_duplicates(subset=['LOCATION'])

    # Merge for origins
    origin_coords = city_coords.rename(columns={
        "LOCATION": "LEGO_ZONE_DESC",
        "LAT": "ORIG_LAT",
        "LON": "ORIG_LON"
    })
    df = df.merge(origin_coords[['LEGO_ZONE_DESC', 'ORIG_LAT', 'ORIG_LON']], 
                 on="LEGO_ZONE_DESC", how="left")
    
    # Merge for destinations
    dest_coords = city_coords.rename(columns={
        "LOCATION": "LEGD_ZONE_DESC",
        "LAT": "DEST_LAT",
        "LON": "DEST_LON"
    })
    df = df.merge(dest_coords[['LEGD_ZONE_DESC', 'DEST_LAT', 'DEST_LON']], 
                 on="LEGD_ZONE_DESC", how="left")
    
    return df

@st.cache_data
def preprocess_tlorder_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    # Convert numeric columns
    df['TOTAL_PAY_SUM'] = pd.to_numeric(df['TOTAL_PAY_SUM'], errors='coerce')
    df['CHARGES'] = pd.to_numeric(df['CHARGES'], errors='coerce')
    df['XCHARGES'] = pd.to_numeric(df['XCHARGES'], errors='coerce')
    df['DISTANCE'] = pd.to_numeric(df['DISTANCE'], errors='coerce')
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

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    # Load and preprocess data
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Convert LS_ACTUAL_DATE to datetime
    legsum_df['LS_ACTUAL_DATE'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE'], format='mixed', errors='coerce')

    # Date range selector
    min_date = legsum_df['LS_ACTUAL_DATE'].min().date()
    max_date = legsum_df['LS_ACTUAL_DATE'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Merge LEGSUM with TLORDER+DRIVERPAY data
    merged_df = legsum_df.merge(tlorder_driverpay_df, 
                               left_on='LS_FREIGHT', 
                               right_on='BILL_NUMBER', 
                               how='left')

    # Filter by date range
    filtered_df = merged_df[
        (merged_df['LS_ACTUAL_DATE'].dt.date >= start_date) &
        (merged_df['LS_ACTUAL_DATE'].dt.date <= end_date)
    ].copy()

    # Calculate financial metrics
    exchange_rate = 1.38
    filtered_df['TOTAL_CHARGE_CAD'] = np.where(
        filtered_df['CURRENCY_CODE'] == 'USD',
        (filtered_df['CHARGES'].fillna(0) + filtered_df['XCHARGES'].fillna(0)) * exchange_rate,
        filtered_df['CHARGES'].fillna(0) + filtered_df['XCHARGES'].fillna(0)
    )

    filtered_df['TOTAL_PAY_SUM'] = filtered_df['TOTAL_PAY_SUM'].fillna(0)
    filtered_df['Profit'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_SUM']
    filtered_df['Revenue per Mile'] = np.where(
        filtered_df['LS_LEG_DIST'] > 0,
        filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['LS_LEG_DIST'],
        0
    )

    # Add truck unit selection
    punit_options = sorted(filtered_df['LS_POWER_UNIT'].astype(str).unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
    
    # Add driver selection
    relevant_drivers = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit]['LS_DRIVER'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver (optional):", options=driver_options)

    # Filter by selections
    view_df = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit].copy()
    if selected_driver != "All":
        view_df = view_df[view_df['LS_DRIVER'] == selected_driver].copy()

    if not view_df.empty:
        # Calculate straight-line distance
        view_df['Straight Distance'] = calculate_haversine(view_df)
        
        # Prepare route summary DataFrame
        route_summary_df = view_df.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[[
            'Route', 'BILL_NUMBER', 'LS_TRIP_NUMBER', 'LS_LEG_DIST', 'LS_MT_LOADED',
            'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'Straight Distance', 'Revenue per Mile',
            'TOTAL_PAY_SUM', 'Profit', 'LS_ACTUAL_DATE', 'LS_LEG_NOTE'
        ]].copy()

        # Format numeric columns
        route_summary_df['TOTAL_CHARGE_CAD'] = route_summary_df['TOTAL_CHARGE_CAD'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        route_summary_df['TOTAL_PAY_SUM'] = route_summary_df['TOTAL_PAY_SUM'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        route_summary_df['Profit'] = route_summary_df['Profit'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        route_summary_df['Revenue per Mile'] = route_summary_df['Revenue per Mile'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        route_summary_df['LS_LEG_DIST'] = route_summary_df['LS_LEG_DIST'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "0.0")
        route_summary_df['Straight Distance'] = route_summary_df['Straight Distance'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "0.0")
        
        # Calculate totals
        totals = pd.DataFrame([{
            'Route': 'TOTALS',
            'BILL_NUMBER': '',
            'LS_TRIP_NUMBER': '',
            'LS_LEG_DIST': f"{view_df['LS_LEG_DIST'].sum():,.1f}",
            'LS_MT_LOADED': '',
            'TOTAL_CHARGE_CAD': f"${view_df['TOTAL_CHARGE_CAD'].sum():,.2f}",
            'Straight Distance': f"{view_df['Straight Distance'].sum():,.1f}",
            'Revenue per Mile': f"${(view_df['TOTAL_CHARGE_CAD'].sum() / view_df['LS_LEG_DIST'].sum() if view_df['LS_LEG_DIST'].sum() > 0 else 0):,.2f}",
            'TOTAL_PAY_SUM': f"${view_df['TOTAL_PAY_SUM'].sum():,.2f}",
            'Profit': f"${view_df['Profit'].sum():,.2f}",
            'LS_ACTUAL_DATE': '',
            'LS_LEG_NOTE': ''
        }])
        
        # Combine with route summary and display
        final_summary = pd.concat([route_summary_df, totals], ignore_index=True)
        st.write("### Route Summary:")
        st.dataframe(final_summary, use_container_width=True)

        # Prepare location aggregates for hover information
        location_aggregates = pd.concat([
            view_df[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_SUM', 'Profit', 'Revenue per Mile']]
            .rename(columns={'LEGO_ZONE_DESC': 'Location'}),
            view_df[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_SUM', 'Profit', 'Revenue per Mile']]
            .rename(columns={'LEGD_ZONE_DESC': 'Location'})
        ]).groupby('Location').agg({
            'TOTAL_CHARGE_CAD': 'sum',
            'LS_LEG_DIST': 'sum',
            'TOTAL_PAY_SUM': 'sum',
            'Profit': 'sum',
            'Revenue per Mile': 'mean'
        }).reset_index()

        # Create the map
        fig = go.Figure()
        
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        # Add markers and routes to the map
        for _, row in view_df.iterrows():
            # Prepare hover text for origin
            origin_agg = location_aggregates[location_aggregates['Location'] == row['LEGO_ZONE_DESC']].iloc[0]
            origin_hover = (
                f"Location: {row['LEGO_ZONE_DESC']}<br>"
                f"Total Charge: ${origin_agg['TOTAL_CHARGE_CAD']:,.2f}<br>"
                f"Distance: {origin_agg['LS_LEG_DIST']:,.1f} miles<br>"
                f"Revenue per Mile: ${origin_agg['Revenue per Mile']:,.2f}<br>"
                f"Driver Pay: ${origin_agg['TOTAL_PAY_SUM']:,.2f}<br>"
                f"Profit: ${origin_agg['Profit']:,.2f}"
            )

            # Add origin marker
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON']],
                lat=[row['ORIG_LAT']],
                mode="markers",
                marker=dict(size=8, color="blue"),
                name="Origin" if not legend_added["Origin"] else None,
                text=row['LEGO_ZONE_DESC'],
                hovertext=origin_hover,
                hoverinfo="text",
                showlegend=not legend_added["Origin"]
            ))
            legend_added["Origin"] = True

            # Prepare hover text for destination
            dest_agg = location_aggregates[location_aggregates['Location'] == row['LEGD_ZONE_DESC']].iloc[0]
            dest_hover = (
                f"Location: {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge: ${dest_agg['TOTAL_CHARGE_CAD']:,.2f}<br>"
                f"Distance: {dest_agg['LS_LEG_DIST']:,.1f} miles<br>"
                f"Revenue per Mile: ${dest_agg['Revenue per Mile']:,.2f}<br>"
                f"Driver Pay: ${dest_agg['TOTAL_PAY_SUM']:,.2f}<br>"
                f"Profit: ${dest_agg['Profit']:,.2f}"
            )

            # Add destination marker
            fig.add_trace(go.Scattergeo(
                lon=[row['DEST_LON']],
                lat=[row['DEST_LAT']],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Destination" if not legend_added["Destination"] else None,
                text=row['LEGD_ZONE_DESC'],
                hovertext=dest_hover,
                hoverinfo="text",
                showlegend=not legend_added["Destination"]
            ))
            legend_added["Destination"] = True

            # Add route line with hover info
            route_hover = (
                f"Route: {row['LEGO_ZONE_DESC']} to {row['LEGD_ZONE_DESC']}<br>"
                f"Trip Number: {row['LS_TRIP_NUMBER']}<br>"
                f"Bill Number: {row['BILL_NUMBER'] if pd.notna(row['BILL_NUMBER']) else 'N/A'}<br>"
                f"Distance: {row['LS_LEG_DIST']:,.1f} miles<br>"
                f"Straight Distance: {row['Straight Distance']:,.1f} miles<br>"
                f"Date: {row['LS_ACTUAL_DATE'].strftime('%Y-%m-%d')}"
            )

            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route" if not legend_added["Route"] else None,
                hovertext=route_hover,
                hoverinfo="text",
                showlegend=not legend_added["Route"]
            ))
            legend_added["Route"] = True

        # Update map layout
        fig.update_layout(
            title=f"Routes for {selected_punit} - {selected_driver}<br>{start_date} to {end_date}",
            geo=dict(
                scope="north america",
                projection_type="mercator",
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 230, 250)',
                showlakes=True,
                lakecolor='rgb(230, 230, 250)'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

        # Display missing locations
        missing_origins = view_df[pd.isna(view_df['ORIG_LAT']) | pd.isna(view_df['ORIG_LON'])][['LEGO_ZONE_DESC']].rename(columns={'LEGO_ZONE_DESC': 'Location'})
        missing_destinations = view_df[pd.isna(view_df['DEST_LAT']) | pd.isna(view_df['DEST_LON'])][['LEGD_ZONE_DESC']].rename(columns={'LEGD_ZONE_DESC': 'Location'})
        missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

        if not missing_locations.empty:
            st.write("### Missing Locations")
            st.dataframe(missing_locations, use_container_width=True)

    else:
        st.warning("No data available for the selected Power Unit and Driver during the specified date range.")

else:
    st.warning("Please upload both the LEGSUM and TLORDER+DRIVERPAY CSV files to proceed.")
