### adding "Find Power Unit" section, postal/zip codes in table

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Initialize global variables for navigation
if "date_index" not in st.session_state:
    st.session_state.date_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_DRIVER, LS_FREIGHT, LS_TRIP_NUMBER, LS_LEG_SEQ, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

Use the following query to generate the required pre-merged TLORDER + DRIVERPAY data:  
SELECT 
    D.BILL_NUMBER, O.CALLNAME, O.ORIGPROV, O.DESTPROV, O.CHARGES, O.XCHARGES, O.DISTANCE,
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
uploaded_tlorder_driverpay_file = st.file_uploader("Upload Pre-Merged TLORDER + DRIVERPAY CSV file", type="csv")

@st.cache_data
def load_city_coordinates():
    city_coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    city_coords.rename(columns={
        "location": "LOCATION",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)

    # Ensure LAT and LON columns are numeric, coercing errors to NaN
    city_coords['LAT'] = pd.to_numeric(city_coords['LAT'], errors='coerce')
    city_coords['LON'] = pd.to_numeric(city_coords['LON'], errors='coerce')
    
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
def preprocess_tlorder_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df['CHARGES'] = pd.to_numeric(df['CHARGES'], errors='coerce')
    df['XCHARGES'] = pd.to_numeric(df['XCHARGES'], errors='coerce')
    df['TOTAL_PAY_SUM'] = pd.to_numeric(df['TOTAL_PAY_SUM'], errors='coerce')
    return df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1 = np.radians(df['LEGO_LAT']), np.radians(df['LEGO_LON'])
    lat2, lon2 = np.radians(df['LEGD_LAT']), np.radians(df['LEGD_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    # Load and preprocess data
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Merge TLORDER+DRIVERPAY data into LEGSUM on BILL_NUMBER
    merged_df = legsum_df.merge(tlorder_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Add a province column for origin zones from LEGSUM and destination provinces from TLORDER
    merged_df['ORIGPROV'] = merged_df['LEGO_ZONE_DESC'].str.split(", ").str[-1]  # Extract province from origin in LEGSUM
    merged_df['DESTPROV'] = merged_df['DESTPROV']  # Use the DESTPROV directly from TLORDER data

    st.header("Power Unit Finder")

    # Dropdown for Customer (CALLNAME) - no "All" option
    callname_options = sorted(merged_df['CALLNAME'].dropna().unique())
    selected_callname = st.selectbox("Select Customer (CALLNAME):", options=callname_options)

    # Filter data for selected customer
    filtered_data = merged_df[merged_df['CALLNAME'] == selected_callname]

    # Dropdowns for Origin and Destination Provinces
    origprov_options = ["All"] + sorted(filtered_data['ORIGPROV'].dropna().unique())
    destprov_options = ["All"] + sorted(filtered_data['DESTPROV'].dropna().unique())

    selected_origprov = st.selectbox("Select Origin Province (ORIGPROV):", options=origprov_options)
    selected_destprov = st.selectbox("Select Destination Province (DESTPROV):", options=destprov_options)

    # Add Start Date and End Date filtering
    st.write("### Select Date Range:")
    if not filtered_data.empty:
        min_date = filtered_data['LS_ACTUAL_DATE'].min().date()
        max_date = filtered_data['LS_ACTUAL_DATE'].max().date()
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        # Apply date range filter
        filtered_data = filtered_data[
            (filtered_data['LS_ACTUAL_DATE'].dt.date >= start_date) &
            (filtered_data['LS_ACTUAL_DATE'].dt.date <= end_date)
        ]

    # Handle combined ORIGPROV and DESTPROV filtering
    if selected_origprov != "All" or selected_destprov != "All":
        # Identify BILL_NUMBERs satisfying the selected ORIGPROV or DESTPROV
        valid_bills = filtered_data[
            ((filtered_data['ORIGPROV'] == selected_origprov) | (selected_origprov == "All")) &
            ((filtered_data['DESTPROV'] == selected_destprov) | (selected_destprov == "All"))
        ]['BILL_NUMBER'].unique()

        # Filter the original dataset to include all legs for these BILL_NUMBERs
        filtered_data = filtered_data[filtered_data['BILL_NUMBER'].isin(valid_bills)]

    if filtered_data.empty:
        st.warning("No results found for the selected criteria.")
    else:
        # Group data by BILL_NUMBER and display separate tables for each
        grouped = filtered_data.groupby('BILL_NUMBER')
        for bill_number, group in grouped:
            # Get the overall ORIGPROV -> DESTPROV for the BILL_NUMBER
            bill_origprov = group['ORIGPROV'].iloc[0]
            bill_destprov = group['DESTPROV'].iloc[0]

            # Display the BILL_NUMBER and its ORIGPROV -> DESTPROV
            st.write(f"### Bill Number: {bill_number} ({bill_origprov} to {bill_destprov})")

            # Sort by LS_ACTUAL_DATE and LS_LEG_SEQ
            group = group.sort_values(by=['LS_ACTUAL_DATE', 'LS_LEG_SEQ'])

            # Create a table showing movements for the bill number
            bill_table = group[['LS_POWER_UNIT', 'LEGO_ZONE_DESC', 'LEGD_ZONE_DESC', 'LS_LEG_SEQ', 'LS_ACTUAL_DATE']].rename(
                columns={
                    'LS_POWER_UNIT': 'Power Unit',
                    'LEGO_ZONE_DESC': 'Origin',
                    'LEGD_ZONE_DESC': 'Destination',
                    'LS_LEG_SEQ': 'Sequence',
                    'LS_ACTUAL_DATE': 'Date'
                }
            )
            bill_table['Date'] = bill_table['Date'].dt.date  # Format date for display

            # Display the table for this bill number
            st.write(bill_table)

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Merge TLORDER+DRIVERPAY data into LEGSUM on BILL_NUMBER
    merged_df = legsum_df.merge(tlorder_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    st.header("Table and Map for Power Unit")

    # Add currency conversion for charges (if applicable)
    exchange_rate = 1.38  # Example USD to CAD conversion rate

    # Ensure CHARGES and XCHARGES are numeric, replacing invalid entries with NaN
    merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
    merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')

    # Set 'CALLNAME' (Customer) to None if there is no BILL_NUMBER
    merged_df['CALLNAME'] = np.where(
        merged_df['BILL_NUMBER'].notna(),  # Only retain Customer if BILL_NUMBER exists
        merged_df['CALLNAME'],
        None  # Set to None otherwise
    )

    # Calculate TOTAL_CHARGE_CAD based on CURRENCY_CODE
    merged_df['TOTAL_CHARGE_CAD'] = np.where(
        merged_df['CURRENCY_CODE'] == 'USD',  # Check if currency is USD
        (merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)) * exchange_rate,  # Convert to CAD
        merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)  # Use original values if not USD
    )

    # Ensure LS_LEG_DIST and DISTANCE are numeric
    merged_df['LS_LEG_DIST'] = pd.to_numeric(merged_df['LS_LEG_DIST'], errors='coerce')  # Leg Distance
    merged_df['DISTANCE'] = pd.to_numeric(merged_df['DISTANCE'], errors='coerce')  # Bill Distance

    # Adjust DISTANCE based on DISTANCE_UNITS (convert KM to miles if applicable)
    merged_df['Bill Distance (miles)'] = np.where(
        merged_df['DISTANCE_UNITS'] == 'KM',
        merged_df['DISTANCE'] * 0.62,  # Convert KM to miles
        merged_df['DISTANCE']  # Use DISTANCE as-is if already in miles
    )

    # Ensure LS_LEG_DIST is positive or assign NaN for invalid values
    merged_df['LS_LEG_DIST'] = np.where(merged_df['LS_LEG_DIST'] > 0, merged_df['LS_LEG_DIST'], np.nan)

    # Calculate Revenue per Mile based on Bill Distance
    merged_df['Revenue per Mile'] = np.where(
        pd.notna(merged_df['TOTAL_CHARGE_CAD']) & pd.notna(merged_df['Bill Distance (miles)']) & (merged_df['Bill Distance (miles)'] > 0),
        merged_df['TOTAL_CHARGE_CAD'] / merged_df['Bill Distance (miles)'],  # Revenue per mile calculation
        np.nan  # Assign NaN if Bill Distance is missing or zero
    )

    # Calculate Profit (CAD) only if TOTAL_CHARGE_CAD is available
    merged_df['Profit (CAD)'] = np.where(
        pd.notna(merged_df['TOTAL_CHARGE_CAD']),
        merged_df['TOTAL_CHARGE_CAD'] - merged_df['TOTAL_PAY_SUM'].fillna(0),  # Calculate Profit
        np.nan  # Assign NaN if TOTAL_CHARGE_CAD is missing
    )

    # Identify locations missing in the coordinates dataset
    missing_origins = merged_df[
        pd.isna(merged_df['LEGO_LAT']) | pd.isna(merged_df['LEGO_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = merged_df[
        pd.isna(merged_df['LEGD_LAT']) | pd.isna(merged_df['LEGD_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Fill missing Straight Distance as np.nan for rows missing coordinates
    merged_df['Straight Distance'] = np.where(
        pd.isna(merged_df['LEGO_LAT']) | pd.isna(merged_df['LEGD_LAT']),
        np.nan,
        calculate_haversine(merged_df)
    )

    # Convert LS_POWER_UNIT to string for consistency
    merged_df['LS_POWER_UNIT'] = merged_df['LS_POWER_UNIT'].astype(str)
    
    # Power Unit selection
    punit_options = sorted(merged_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
    
    # Filter data for the selected Power Unit
    filtered_view = merged_df[merged_df['LS_POWER_UNIT'] == selected_punit].copy()
    
    # Driver selection (optional)
    relevant_drivers = filtered_view['LS_DRIVER'].dropna().unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['LS_DRIVER'] == selected_driver].copy()
    
    # Date Range Filtering
    st.write("### Select Date Range:")
    min_date = filtered_view['LS_ACTUAL_DATE'].min().date()
    max_date = filtered_view['LS_ACTUAL_DATE'].max().date()
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Filter by selected date range
    filtered_view = filtered_view[
        (filtered_view['LS_ACTUAL_DATE'].dt.date >= start_date) &
        (filtered_view['LS_ACTUAL_DATE'].dt.date <= end_date)
    ].copy()
    
    if filtered_view.empty:
        st.warning("No data available for the selected criteria.")
    else:
        # Deduplicate rows based on 'LS_POWER_UNIT', 'Route', and 'LS_ACTUAL_DATE'
        filtered_view['Route'] = filtered_view['LEGO_ZONE_DESC'] + " to " + filtered_view['LEGD_ZONE_DESC']
        filtered_view = filtered_view.drop_duplicates(subset=['LS_POWER_UNIT', 'Route', 'LS_ACTUAL_DATE'], keep='first')
    
        # Filter by selected date range
        filtered_view = filtered_view[
            (filtered_view['LS_ACTUAL_DATE'].dt.date >= start_date) &
            (filtered_view['LS_ACTUAL_DATE'].dt.date <= end_date)
        ].copy()
        
        if filtered_view.empty:
            st.warning("No data available for the selected criteria.")
        else:
            # Deduplicate rows based on 'LS_POWER_UNIT', 'Route', and 'LS_ACTUAL_DATE'
            filtered_view['Route'] = filtered_view['LEGO_ZONE_DESC'] + " to " + filtered_view['LEGD_ZONE_DESC']
            filtered_view = filtered_view.drop_duplicates(subset=['LS_POWER_UNIT', 'Route', 'LS_ACTUAL_DATE'], keep='first')
            
            # Sort by LS_ACTUAL_DATE (primary) and LS_LEG_SEQ (secondary)
            filtered_view = filtered_view.sort_values(by=['LS_ACTUAL_DATE', 'LS_LEG_SEQ'])
            
            # Assign colors for alternating rows by day
            filtered_view['Day_Group'] = filtered_view['LS_ACTUAL_DATE'].dt.date
            unique_days = list(filtered_view['Day_Group'].unique())
            day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
            filtered_view['Highlight'] = filtered_view['Day_Group'].map(day_colors)
            
            # Add calculated fields
            filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_SUM']
            filtered_view['Revenue per Mile'] = np.where(
                pd.notna(filtered_view['Bill Distance (miles)']) & (filtered_view['Bill Distance (miles)'] > 0),
                filtered_view['TOTAL_CHARGE_CAD'] / filtered_view['Bill Distance (miles)'],
                np.nan
            )
        
            # Create the route summary DataFrame
            route_summary_df = filtered_view[
                [
                    "Route", "LS_FREIGHT", "CALLNAME", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Bill Distance (miles)",
                    "Revenue per Mile", "LS_DRIVER", "TOTAL_PAY_SUM", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
                ]
            ].rename(columns={
                "LS_FREIGHT": "BILL_NUMBER",
                "CALLNAME": "Customer",
                "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
                "LS_LEG_DIST": "Leg Distance (miles)",
                "Bill Distance (miles)": "Bill Distance (miles)",
                "TOTAL_PAY_SUM": "Driver Pay (CAD)"
            })
    
        # Add calculated fields
        filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_SUM']
        filtered_view['Revenue per Mile'] = np.where(
            pd.notna(filtered_view['Bill Distance (miles)']) & (filtered_view['Bill Distance (miles)'] > 0),
            filtered_view['TOTAL_CHARGE_CAD'] / filtered_view['Bill Distance (miles)'],
            np.nan
        )
    
        # Create the route summary DataFrame
        route_summary_df = filtered_view[
            [
                "Route", "LS_FREIGHT", "CALLNAME", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Bill Distance (miles)", 
                "Revenue per Mile", "LS_DRIVER", "TOTAL_PAY_SUM", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
            ]
        ].rename(columns={
            "LS_FREIGHT": "BILL_NUMBER",
            "CALLNAME": "Customer",
            "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
            "LS_LEG_DIST": "Leg Distance (miles)",
            "Bill Distance (miles)": "Bill Distance (miles)",
            "TOTAL_PAY_SUM": "Driver Pay (CAD)"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Customer": "",
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Leg Distance (miles)": route_summary_df["Leg Distance (miles)"].sum(),
            "Bill Distance (miles)": route_summary_df["Bill Distance (miles)"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Bill Distance (miles)"].sum()
            if route_summary_df["Bill Distance (miles)"].sum() != 0 else 0,
            "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "LS_ACTUAL_DATE": "",
            "LS_LEG_NOTE": "",
            "Highlight": None
        }])
        
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
        
        # Format currency and numeric columns
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
            filtered_view[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'Bill Distance (miles)', 'TOTAL_PAY_SUM']].rename(
                columns={'LEGO_ZONE_DESC': 'Location'}
            ),
            filtered_view[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'Bill Distance (miles)', 'TOTAL_PAY_SUM']].rename(
                columns={'LEGD_ZONE_DESC': 'Location'}
            )
        ], ignore_index=True)
    
        # Aggregate data by location
        location_aggregates = location_aggregates.groupby(['Location'], as_index=False).agg({
            'TOTAL_CHARGE_CAD': 'sum',
            'Bill Distance (miles)': 'sum',
            'TOTAL_PAY_SUM': 'sum'
        })
    
        location_aggregates['Revenue per Mile'] = location_aggregates['TOTAL_CHARGE_CAD'] / location_aggregates['Bill Distance (miles)']
        location_aggregates['Profit (CAD)'] = location_aggregates['TOTAL_CHARGE_CAD'] - location_aggregates['TOTAL_PAY_SUM']
    
        # Function to fetch aggregate values for a location
        def get_location_aggregates(location):
            match = location_aggregates[location_aggregates['Location'] == location]
            if not match.empty:
                total_charge = match['TOTAL_CHARGE_CAD'].iloc[0]
                bill_distance = match['Bill Distance (miles)'].iloc[0]
                driver_pay = match['TOTAL_PAY_SUM'].iloc[0]
                profit = match['Profit (CAD)'].iloc[0]
                rpm = match['Revenue per Mile'].iloc[0]
                return total_charge, bill_distance, driver_pay, profit, rpm
            return 0, 0, 0, 0, 0

        # Generate the map
        fig = go.Figure()
        
        # Identify missing locations
        missing_origins = filtered_view[
            pd.isna(filtered_view['LEGO_LAT']) | pd.isna(filtered_view['LEGO_LON'])
        ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})
        
        missing_destinations = filtered_view[
            pd.isna(filtered_view['LEGD_LAT']) | pd.isna(filtered_view['LEGD_LON'])
        ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})
        
        missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()
        missing_location_set = set(missing_locations['Location'])
        
        # Track sequence of city appearance for labeling
        location_sequence = {}
        label_counter = 1
        
        # Number all origins in order
        for _, row in filtered_view.iterrows():
            if row['LEGO_ZONE_DESC'] not in missing_location_set:
                if row['LEGO_ZONE_DESC'] not in location_sequence:
                    location_sequence[row['LEGO_ZONE_DESC']] = []
                location_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
                label_counter += 1
        
        # Add the final destination
        final_destination = filtered_view.iloc[-1]['LEGD_ZONE_DESC']
        if final_destination not in missing_location_set:
            if final_destination not in location_sequence:
                location_sequence[final_destination] = []
            location_sequence[final_destination].append(label_counter)
        
        # Initialize legend flags
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        # Loop through filtered data to create map elements
        for _, row in filtered_view.iterrows():
            origin_sequence = ", ".join(map(str, location_sequence.get(row['LEGO_ZONE_DESC'], [])))
            destination_sequence = ", ".join(map(str, location_sequence.get(row['LEGD_ZONE_DESC'], [])))
        
            # Get aggregated values for origin location
            total_charge, bill_distance, driver_pay, profit, rpm = get_location_aggregates(row['LEGO_ZONE_DESC'])
            hover_origin_text = (
                f"Location: {row['LEGO_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${total_charge:,.2f}<br>"
                f"Bill Distance (miles): {bill_distance:,.1f}<br>"
                f"Revenue per Mile: ${rpm:,.2f}<br>"
                f"Driver Pay (CAD): ${driver_pay:,.2f}<br>"
                f"Profit (CAD): ${profit:,.2f}"
            )
        
            # Add origin marker
            if row['LEGO_ZONE_DESC'] not in missing_location_set:
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
            total_charge, bill_distance, driver_pay, profit, rpm = get_location_aggregates(row['LEGD_ZONE_DESC'])
            hover_dest_text = (
                f"Location: {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge (CAD): ${total_charge:,.2f}<br>"
                f"Bill Distance (miles): {bill_distance:,.1f}<br>"
                f"Revenue per Mile: ${rpm:,.2f}<br>"
                f"Driver Pay (CAD): ${driver_pay:,.2f}<br>"
                f"Profit (CAD): ${profit:,.2f}"
            )
        
            # Add destination marker
            if row['LEGD_ZONE_DESC'] not in missing_location_set:
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
        
        # Configure map layout
        fig.update_layout(
            title=f"Routes from {start_date} to {end_date} - Power Unit: {selected_punit}, Driver ID: {selected_driver}",
            geo=dict(
                scope="north america",
                projection_type="mercator",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(217, 217, 217)",
                countrycolor="rgb(217, 217, 217)"
            )
        )
        
        st.plotly_chart(fig)
        
        # Display locations missing coordinates
        if not missing_locations.empty:
            st.write("### Locations Missing Coordinates")
            st.dataframe(missing_locations, use_container_width=True)

else:
    st.warning("Please upload LEGSUM and TLORDER+DRIVERPAY CSV files to proceed.")
    
