import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer by Month (LEGSUM)")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_FREIGHT, LS_TRIP_NUMBER, LS_TO_ZONE, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

Use the following query to generate the required TLORDER data:  
SELECT BILL_NUMBER, DETAIL_LINE_ID, CALLNAME, ORIGCITY, ORIGPROV, DESTCITY, DESTPROV, PICK_UP_PUNIT, DELIVERY_PUNIT, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE, PICK_UP_BY, DELIVER_BY  
FROM TLORDER WHERE "PICK_UP_BY" BETWEEN 'X' AND 'Y';  

Use the following query to generate the required DRIVERPAY data:  
SELECT BILL_NUMBER, PAY_ID, DRIVER_ID, PAY_DESCRIPTION, FB_TOTAL_CHARGES, CURRENCY_CODE, TOTAL_PAY_AMT, PAID_DATE, DATE_TRANS  
FROM DRIVERPAY WHERE "PAID_DATE" BETWEEN 'X' AND 'Y';  

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file (optional, for merging BILL_NUMBER data)", type="csv")
uploaded_driverpay_file = st.file_uploader("Upload DRIVERPAY CSV file (optional, for merging pay data)", type="csv")

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
  df['LS_ACTUAL_DATE'] = pd.to_datetime(df['LS_ACTUAL_DATE'], errors='coerce')

  # Standardize location names
  city_coords['LOCATION'] = city_coords['LOCATION'].str.strip().str.upper()
  df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].str.strip().str.upper()
  df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].str.strip().str.upper()
  
  # Merge for LEGO_ZONE_DESC
  lego_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "LEGO_LAT", "LON": "LEGO_LON"})
  df = df.merge(lego_coords, on="LEGO_ZONE_DESC", how="left")
  
  # Merge for LEGD_ZONE_DESC
  legd_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "LEGD_LAT", "LON": "LEGD_LON"})
  df = df.merge(legd_coords, on="LEGD_ZONE_DESC", how="left")
  
  return df

@st.cache_data
def filter_and_enrich_locations(df, city_coords):
    # Standardize location names in city_coords
    city_coords['LOCATION'] = city_coords['LOCATION'].str.strip().str.upper()

    # Extract unique LEGO_ZONE_DESC and LEGD_ZONE_DESC from the dataset
    relevant_origins = df[['LEGO_ZONE_DESC']].drop_duplicates()
    relevant_destinations = df[['LEGD_ZONE_DESC']].drop_duplicates()

    relevant_locations = pd.concat([relevant_origins.rename(columns={'LEGO_ZONE_DESC': 'LOCATION'}),
                                     relevant_destinations.rename(columns={'LEGD_ZONE_DESC': 'LOCATION'})]
                                   ).drop_duplicates()

    # Merge with city_coords to enrich relevant locations with latitude and longitude
    enriched_locations = relevant_locations.merge(city_coords, on="LOCATION", how="left")

    # Remove duplicates from the final enriched locations
    enriched_locations = enriched_locations.drop_duplicates()

    return enriched_locations

@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_AMT'] = pd.to_numeric(df['TOTAL_PAY_AMT'], errors='coerce')
    driver_pay_agg = df.groupby('BILL_NUMBER').agg({
        'TOTAL_PAY_AMT': 'sum',
        'DRIVER_ID': 'first'
    }).reset_index()
    return driver_pay_agg

@st.cache_data
def calculate_haversine(df):
    """
    Calculate the great-circle distance (haversine formula) between origin and destination coordinates.
    Applies to LEGO_LAT, LEGO_LON (origin) and LEGD_LAT, LEGD_LON (destination) in LEGSUM.
    """
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1 = np.radians(df['LEGO_LAT']), np.radians(df['LEGO_LON'])
    lat2, lon2 = np.radians(df['LEGD_LAT']), np.radians(df['LEGD_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)

    # Optional: Add DRIVERPAY data if uploaded
    if uploaded_driverpay_file:
        driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)
        legsum_df = legsum_df.merge(driver_pay_agg, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Add currency conversion for charges (if applicable)
    exchange_rate = 1.38
    legsum_df['CHARGES'] = pd.to_numeric(legsum_df.get('CHARGES', 0), errors='coerce')
    legsum_df['TOTAL_CHARGE_CAD'] = legsum_df['CHARGES'] * exchange_rate

    legsum_df['TOTAL_CHARGE_CAD'] = np.where(
        pd.notna(legsum_df['BILL_NUMBER']),
        legsum_df['CHARGES'] * 1.38,  # Assuming 'CHARGES' is in USD; apply conversion
        None  # Set to None if BILL_NUMBER is missing
    )

    # Calculate Revenue per Mile and Profit
    legsum_df['Revenue per Mile'] = legsum_df['TOTAL_CHARGE_CAD'] / legsum_df['LS_LEG_DIST']
    legsum_df['Profit (CAD)'] = legsum_df['TOTAL_CHARGE_CAD'] - legsum_df['TOTAL_PAY_AMT'].fillna(0)

    # Add a Month column for grouping
    legsum_df['Month'] = legsum_df['LS_ACTUAL_DATE'].dt.to_period('M')

    # Identify locations missing in the coordinates dataset
    missing_origins = legsum_df[
        pd.isna(legsum_df['LEGO_LAT']) | pd.isna(legsum_df['LEGO_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = legsum_df[
        pd.isna(legsum_df['LEGD_LAT']) | pd.isna(legsum_df['LEGD_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Fill missing Straight Distance as np.nan for rows missing coordinates
    legsum_df['Straight Distance'] = np.where(
        pd.isna(legsum_df['LEGO_LAT']) | pd.isna(legsum_df['LEGD_LAT']),
        np.nan,
        calculate_haversine(legsum_df)
    )

    legsum_df['LS_POWER_UNIT'] = legsum_df['LS_POWER_UNIT'].astype(str)
    punit_options = sorted(legsum_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
    relevant_drivers = legsum_df[legsum_df['LS_POWER_UNIT'] == selected_punit]['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    filtered_view = legsum_df[legsum_df['LS_POWER_UNIT'] == selected_punit].copy()
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()
    
    months = sorted(filtered_view['Month'].unique())
    if len(months) == 0:
        st.warning("No data available for the selected Power Unit and Driver ID.")
    else:
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()
    
    if not month_data.empty:
        # Assign colors for alternating rows by day
        month_data = month_data.sort_values(by='LS_ACTUAL_DATE')
        month_data['Day_Group'] = month_data['LS_ACTUAL_DATE'].dt.date
        unique_days = list(month_data['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        month_data['Highlight'] = month_data['Day_Group'].map(day_colors)
    
        # Create the route summary DataFrame
        month_data['Profit (CAD)'] = month_data['TOTAL_CHARGE_CAD'] - month_data['TOTAL_PAY_AMT']
    
        route_summary_df = month_data.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[[  # Include "Highlight" for styling
            "Route", "LS_FREIGHT", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Straight Distance",
            "Revenue per Mile", "DRIVER_ID", "TOTAL_PAY_AMT", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight"
        ]].rename(columns={
            "LS_FREIGHT": "BILL_NUMBER",
            "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
            "LS_LEG_DIST": "Distance (miles)",
            "Straight Distance": "Straight Distance (miles)",
            "TOTAL_PAY_AMT": "Driver Pay (CAD)"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Distance (miles)": route_summary_df["Distance (miles)"].sum(),
            "Straight Distance (miles)": route_summary_df["Straight Distance (miles)"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Distance (miles)"].sum()
            if route_summary_df["Distance (miles)"].sum() != 0 else 0,
            "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "LS_ACTUAL_DATE": "",
            "LS_LEG_NOTE": "",
            "Highlight": None
        }])
    
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)

        # Format currency columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
            route_summary_df[col] = route_summary_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )
        
        # Define row styling
        def highlight_rows(row):
            if row['Route'] == "Grand Totals":
                return ['background-color: #f7c8c8'] * len(row)
            elif row['Highlight'] == 1:
                return ['background-color: #c8e0f7'] * len(row)
            else:
                return ['background-color: #f7f7c8'] * len(row)
        
        styled_route_summary = route_summary_df.style.apply(highlight_rows, axis=1)
        st.write("Route Summary:")
        st.dataframe(styled_route_summary, use_container_width=True)
        
        # Combine origins and destinations for aggregated totals
        location_aggregates = pd.concat([
            month_data[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGO_ZONE_DESC': 'Location'}
            ),
            month_data[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGD_ZONE_DESC': 'Location'}
            )
        ], ignore_index=True)
        
        # Clean and aggregate the combined data
        location_aggregates = location_aggregates.groupby(['Location'], as_index=False).agg({
            'TOTAL_CHARGE_CAD': 'sum',
            'LS_LEG_DIST': 'sum',
            'TOTAL_PAY_AMT': 'sum'
        })
        
        location_aggregates['Revenue per Mile'] = location_aggregates['TOTAL_CHARGE_CAD'] / location_aggregates['LS_LEG_DIST']
        location_aggregates['Profit (CAD)'] = location_aggregates['TOTAL_CHARGE_CAD'] - location_aggregates['TOTAL_PAY_AMT'].fillna(0)
        
        # Function to fetch aggregate values for a location
        def get_location_aggregates(location):
            match = location_aggregates[location_aggregates['Location'] == location]
            if not match.empty:
                total_charge = match['TOTAL_CHARGE_CAD'].iloc[0]
                distance = match['LS_LEG_DIST'].iloc[0]
                driver_pay = match['TOTAL_PAY_AMT'].iloc[0]
                profit = match['Profit (CAD)'].iloc[0]
                rpm = match['Revenue per Mile'].iloc[0]
                return total_charge, distance, driver_pay, profit, rpm
            return 0, 0, 0, 0, 0

        # Generate the map
        fig = go.Figure()
        
        # Track sequence of city appearance
        location_sequence = {location: [] for location in set(month_data['LEGO_ZONE_DESC']).union(month_data['LEGD_ZONE_DESC'])}
        label_counter = 1
        for _, row in month_data.iterrows():
            location_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
            label_counter += 1
            location_sequence[row['LEGD_ZONE_DESC']].append(label_counter)
            label_counter += 1
        
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        for _, row in month_data.iterrows():
            origin_sequence = ", ".join(map(str, location_sequence[row['LEGO_ZONE_DESC']]))
            destination_sequence = ", ".join(map(str, location_sequence[row['LEGD_ZONE_DESC']]))
        
            # Get aggregated values for origin location
            total_charge, distance, driver_pay, profit, rpm = get_location_aggregates(row['LEGO_ZONE_DESC'])
            hover_origin_text = (
                f"Location: {row['LEGO_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${total_charge:,.2f}<br>"
                f"Distance (miles): {distance:,.1f}<br>"
                f"Revenue per Mile: ${rpm:,.2f}<br>"
                f"Driver Pay (CAD): ${driver_pay:,.2f}<br>"
                f"Profit (CAD): ${profit:,.2f}"
            )
        
            # Add origin marker
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON']],
                lat=[row['LEGO_LAT']],
                mode="markers+text",
                marker=dict(size=8, color="blue"),
                text=origin_sequence,
                textposition="top right",
                name="Origin" if not legend_added["Origin"] else None,
                hoverinfo="text",
                hovertext=hover_origin_text,
                showlegend=not legend_added["Origin"]
            ))
            legend_added["Origin"] = True
        
            # Get aggregated values for destination location
            total_charge, distance, driver_pay, profit, rpm = get_location_aggregates(row['LEGD_ZONE_DESC'])
            hover_dest_text = (
                f"Location: {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${total_charge:,.2f}<br>"
                f"Distance (miles): {distance:,.1f}<br>"
                f"Revenue per Mile: ${rpm:,.2f}<br>"
                f"Driver Pay (CAD): ${driver_pay:,.2f}<br>"
                f"Profit (CAD): ${profit:,.2f}"
            )
        
            # Add destination marker
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGD_LON']],
                lat=[row['LEGD_LAT']],
                mode="markers+text",
                marker=dict(size=8, color="red"),
                text=destination_sequence,
                textposition="top right",
                name="Destination" if not legend_added["Destination"] else None,
                hoverinfo="text",
                hovertext=hover_dest_text,
                showlegend=not legend_added["Destination"]
            ))
            legend_added["Destination"] = True
        
            # Add route line
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON'], row['LEGD_LON']],
                lat=[row['LEGO_LAT'], row['LEGD_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route" if not legend_added["Route"] else None,
                hoverinfo="skip",
                showlegend=not legend_added["Route"]
            ))
            legend_added["Route"] = True
        
        fig.update_layout(
            title=f"Routes for {selected_month} - Power Unit: {selected_punit}, Driver ID: {selected_driver}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)

        # Display locations missing coordinates relevant to the selection
        relevant_missing_origins = month_data[
            (pd.isna(month_data['LEGO_LAT']) | pd.isna(month_data['LEGO_LON']))
        ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})
        
        relevant_missing_destinations = month_data[
            (pd.isna(month_data['LEGD_LAT']) | pd.isna(month_data['LEGD_LON']))
        ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})
        
        relevant_missing_locations = pd.concat([relevant_missing_origins, relevant_missing_destinations]).drop_duplicates()
        
        if not relevant_missing_locations.empty:
            st.write("### Locations Missing Coordinates")
            st.dataframe(relevant_missing_locations, use_container_width=True)
        
    else:
        st.warning("No data available for the selected Power Unit and Driver ID.")

else:
    st.warning("Please upload LEGSUM, TLORDER and DRIVERPAY CSV files to proceed.")
