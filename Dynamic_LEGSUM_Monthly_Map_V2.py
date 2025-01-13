### fixing total charge and driver pay since they're 2-5x

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Initialize global variables for navigation
if "date_index" not in st.session_state:
    st.session_state.date_index = 0

# Streamlit App Title and Instructions
st.title("Power Unit Trip Map Viewer for Month")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_DRIVER, LS_FREIGHT, LS_TRIP_NUMBER, LS_LEG_SEQ, LS_FROM_ZONE, LS_TO_ZONE, 
       LEGO_ZONE_DESC, LEGD_ZONE_DESC, LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
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

Recommendation: add 2 weeks to Y for TLORDER + DRIVERPAY data to account for delayed driver pay.

Save the query results as CSV files and upload them below to visualize the data.
""")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_driverpay_file = st.file_uploader("Upload Pre-Merged TLORDER + DRIVERPAY CSV file", type="csv")
# File upload for the new ISAAC reports
uploaded_isaac_owner_ops_file = st.file_uploader("Upload ISAAC Owner Ops Fuel Report (Excel file)", type="xlsx")
uploaded_isaac_company_trucks_file = st.file_uploader("Upload ISAAC Company Trucks Fuel Report (Excel file)", type="xlsx")

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
def preprocess_new_isaac_fuel(file):
    """
    Preprocess the new ISAAC Fuel Report.
    Extracts and aggregates total fuel quantity (liters) for each vehicle number.
    """
    # Load the Excel file
    fuel_df = pd.read_excel(file)

    # Ensure column names are standardized
    fuel_df.rename(columns={
        "Unit": "VEHICLE_NO",
        "Fuel (Litre)": "FUEL_QUANTITY_L"
    }, inplace=True)

    # Retain only relevant columns
    fuel_df = fuel_df[["VEHICLE_NO", "FUEL_QUANTITY_L"]]

    # Ensure data is clean and consistent
    fuel_df['VEHICLE_NO'] = fuel_df['VEHICLE_NO'].astype(str).str.strip()
    fuel_df['FUEL_QUANTITY_L'] = pd.to_numeric(fuel_df['FUEL_QUANTITY_L'], errors='coerce').fillna(0)

    # Aggregate total fuel quantity per vehicle
    fuel_aggregated = fuel_df.groupby('VEHICLE_NO', as_index=False).agg({
        'FUEL_QUANTITY_L': 'sum'
    })

    return fuel_aggregated

# Preprocess the ISAAC reports if both are uploaded
if uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
    isaac_owner_ops_df = preprocess_new_isaac_fuel(uploaded_isaac_owner_ops_file)
    isaac_company_trucks_df = preprocess_new_isaac_fuel(uploaded_isaac_company_trucks_file)

    # Combine the two reports into a single DataFrame
    isaac_combined_fuel_df = pd.concat([isaac_owner_ops_df, isaac_company_trucks_df], ignore_index=True)
else:
    isaac_combined_fuel_df = None

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
    if uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
        # Combine the two ISAAC Fuel Reports
        isaac_combined_fuel_df = pd.concat(
            [preprocess_new_isaac_fuel(uploaded_isaac_owner_ops_file), preprocess_new_isaac_fuel(uploaded_isaac_company_trucks_file)],
            ignore_index=True
        )
    else:
        isaac_combined_fuel_df = None

    # Merge TLORDER+DRIVERPAY data into LEGSUM on BILL_NUMBER
    merged_df = legsum_df.merge(
        tlorder_driverpay_df,
        left_on='LS_FREIGHT',
        right_on='BILL_NUMBER',
        how='left'
    )

    # Merge ISAAC Fuel Report into the merged_df on LS_POWER_UNIT and VEHICLE_NO
    merged_df = merged_df.merge(
        isaac_combined_fuel_df,
        left_on='LS_POWER_UNIT',
        right_on='VEHICLE_NO',
        how='left'
    )

    # drop VEHICLE_NO from the final DataFrame
    merged_df.drop(columns=['VEHICLE_NO'], inplace=True, errors='ignore')

    # Add a province column for both origin and destination from TLORDER data
    merged_df['ORIGPROV'] = merged_df['ORIGPROV']  # Use ORIGPROV from TLORDER
    merged_df['DESTPROV'] = merged_df['DESTPROV']  # Use DESTPROV from TLORDER

    with st.expander("Power Unit Finder by Customer"):

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
    
        # Add Start Date and End Date filtering with unique keys
        st.write("### Select Date Range:")
        if not filtered_data.empty:
            min_date = filtered_data['LS_ACTUAL_DATE'].min().date()
            max_date = filtered_data['LS_ACTUAL_DATE'].max().date()
        
            # Use unique keys for each date_input
            start_date = st.date_input(
                "Start Date", 
                value=min_date, 
                min_value=min_date, 
                max_value=max_date, 
                key=f"start_date_{selected_callname}"
            )
            end_date = st.date_input(
                "End Date", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date, 
                key=f"end_date_{selected_callname}"
            )
        
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


if uploaded_legsum_file and uploaded_tlorder_driverpay_file and uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
    # Preprocess and merge data
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)
    
    # Preprocess both ISAAC Fuel Reports
    owner_ops_fuel_df = preprocess_new_isaac_fuel(uploaded_isaac_owner_ops_file)
    company_trucks_fuel_df = preprocess_new_isaac_fuel(uploaded_isaac_company_trucks_file)

    # Combine the two ISAAC Fuel Reports
    isaac_combined_fuel_df = pd.concat([owner_ops_fuel_df, company_trucks_fuel_df], ignore_index=True)

    # Merge LEGSUM with TLORDER + DRIVERPAY
    merged_df = legsum_df.merge(
        tlorder_driverpay_df,
        left_on='LS_FREIGHT',
        right_on='BILL_NUMBER',
        how='left'
    )

    # Merge with the combined ISAAC Fuel Report
    merged_df = merged_df.merge(
        isaac_combined_fuel_df,
        left_on='LS_POWER_UNIT',
        right_on='VEHICLE_NO',
        how='left'
    )
    merged_df.drop(columns=['VEHICLE_NO'], inplace=True, errors='ignore')

    # Extract the month and year from the dataset
    merged_df['LS_ACTUAL_DATE'] = pd.to_datetime(merged_df['LS_ACTUAL_DATE'], errors='coerce')
    if not merged_df['LS_ACTUAL_DATE'].isna().all():
        # Get the month and year from the data
        month_name = merged_df['LS_ACTUAL_DATE'].dt.month_name().iloc[0]
        year = merged_df['LS_ACTUAL_DATE'].dt.year.iloc[0]
        month_year_title = f"{month_name} {year}"
    else:
        month_year_title = "Unknown Month"

    with st.expander("Table and Map for Power Unit"):

        # Add currency conversion for charges (if applicable)
        exchange_rate = 1.38  # Example USD to CAD conversion rate
    
        # Ensure CHARGES, XCHARGES, and DISTANCE are numeric, replacing invalid entries with NaN
        merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
        merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')
        merged_df['DISTANCE'] = pd.to_numeric(merged_df['DISTANCE'], errors='coerce')
    
        # Set 'CALLNAME' (Customer) to None if there is no BILL_NUMBER
        merged_df['CALLNAME'] = np.where(
            merged_df['BILL_NUMBER'].notna(),  # Only retain Customer if BILL_NUMBER exists
            merged_df['CALLNAME'],
            None  # Set to None otherwise
        )
    
        # Add currency conversion for charges (if applicable)
        exchange_rate = 1.38  # Example USD to CAD conversion rate
        
        # Ensure CHARGES, XCHARGES, and DISTANCE are numeric
        merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
        merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')
        merged_df['DISTANCE'] = pd.to_numeric(merged_df['DISTANCE'], errors='coerce')
        
        # Calculate TOTAL_CHARGE_CAD at the row level
        merged_df['ROW_TOTAL_CHARGE_CAD'] = np.where(
            merged_df['CURRENCY_CODE'] == 'USD',
            (merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)) * exchange_rate,
            merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)
        )
        
        # Deduplicate merged_df at the BILL_NUMBER level to avoid inflated totals
        deduplicated_df = merged_df.drop_duplicates(subset=['BILL_NUMBER'])
        
        # Aggregate charges, distance, and driver pay by BILL_NUMBER
        bill_aggregates = deduplicated_df.groupby('BILL_NUMBER').agg({
            'ROW_TOTAL_CHARGE_CAD': 'sum',  # Sum ROW_TOTAL_CHARGE_CAD per BILL_NUMBER
            'DISTANCE': 'sum',  # Sum DISTANCE per BILL_NUMBER
            'TOTAL_PAY_SUM': 'sum',  # Sum Driver Pay
        }).reset_index().rename(columns={
            'ROW_TOTAL_CHARGE_CAD': 'TOTAL_CHARGE_CAD',
            'DISTANCE': 'AGGREGATED_DISTANCE',
            'TOTAL_PAY_SUM': 'AGGREGATED_PAY_SUM'
        })
        
        # Merge aggregated totals back into merged_df
        merged_df = merged_df.merge(
            bill_aggregates[['BILL_NUMBER', 'TOTAL_CHARGE_CAD', 'AGGREGATED_DISTANCE', 'AGGREGATED_PAY_SUM']],
            on='BILL_NUMBER',
            how='left'
        )
        
        # Avoid overwriting original DISTANCE/Driver Pay columns; create aggregated fields
        merged_df['Bill Distance (miles)'] = np.where(
            merged_df['DISTANCE_UNITS'] == 'KM',
            merged_df['AGGREGATED_DISTANCE'] * 0.62,  # Convert KM to miles
            merged_df['AGGREGATED_DISTANCE']
        )
        
        # Assign Driver Pay (CAD) from aggregated totals
        merged_df['Driver Pay (CAD)'] = merged_df['AGGREGATED_PAY_SUM']
    
        # Adjust DISTANCE based on DISTANCE_UNITS (convert KM to miles if applicable)
        merged_df['Bill Distance (miles)'] = np.where(
            merged_df['DISTANCE_UNITS'] == 'KM',
            merged_df['DISTANCE'] * 0.62,  # Convert aggregated DISTANCE from KM to miles
            merged_df['DISTANCE']  # Use aggregated DISTANCE as-is if already in miles
        )
    
        # Ensure LS_LEG_DIST is positive or assign NaN for invalid values
        merged_df['LS_LEG_DIST'] = np.where(
            merged_df['LS_LEG_DIST'] > 0, 
            merged_df['LS_LEG_DIST'], 
            np.nan  # Assign NaN if LS_LEG_DIST is invalid or negative
        )
        
        # Calculate Revenue per Mile based on aggregated Bill Distance
        merged_df['Revenue per Mile'] = np.where(
            pd.notna(merged_df['TOTAL_CHARGE_CAD']) & 
            pd.notna(merged_df['Bill Distance (miles)']) & 
            (merged_df['Bill Distance (miles)'] > 0),
            merged_df['TOTAL_CHARGE_CAD'] / merged_df['Bill Distance (miles)'],  # Revenue per mile calculation
            np.nan  # Assign NaN if Bill Distance is missing or zero
        )
        
        # Assign Driver Pay (CAD) based on AGGREGATED_PAY_SUM
        merged_df['Driver Pay (CAD)'] = np.where(
            pd.notna(merged_df['BILL_NUMBER']),  # Only assign if BILL_NUMBER exists
            merged_df['AGGREGATED_PAY_SUM'].fillna(0),  # Use aggregated TOTAL_PAY_SUM for valid BILL_NUMBERs
            0  # Otherwise, assign 0
        )
        
        # Calculate Profit (CAD) based on aggregated values
        merged_df['Profit (CAD)'] = np.where(
            pd.notna(merged_df['TOTAL_CHARGE_CAD']) & 
            pd.notna(merged_df['Driver Pay (CAD)']),
            merged_df['TOTAL_CHARGE_CAD'] - merged_df['Driver Pay (CAD)'],  # Profit = Total Charge - Driver Pay
            np.nan  # Assign NaN if TOTAL_CHARGE_CAD or Driver Pay is missing
        )
        
        # Format the Bill Distance (miles) to 1 decimal place for consistency
        merged_df['Bill Distance (miles)'] = merged_df['Bill Distance (miles)'].apply(
            lambda x: round(x, 1) if pd.notna(x) else np.nan
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
                        "Revenue per Mile", "LS_DRIVER", "Driver Pay (CAD)", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
                    ]
                ].rename(columns={
                    "LS_FREIGHT": "BILL_NUMBER",
                    "CALLNAME": "Customer",
                    "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
                    "LS_LEG_DIST": "Leg Distance (miles)",
                    "Bill Distance (miles)": "Bill Distance (miles)"
                })
        
            # Add calculated fields
            filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['Driver Pay (CAD)']
            filtered_view['Revenue per Mile'] = np.where(
                pd.notna(filtered_view['Bill Distance (miles)']) & (filtered_view['Bill Distance (miles)'] > 0),
                filtered_view['TOTAL_CHARGE_CAD'] / filtered_view['Bill Distance (miles)'],
                np.nan
            )
            
            # Create the route summary DataFrame
            route_summary_df = filtered_view[
                [
                    "Route", "LS_FROM_ZONE", "LS_TO_ZONE", "LS_FREIGHT", "CALLNAME", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", 
                    "Bill Distance (miles)", "Revenue per Mile", "LS_DRIVER", "Driver Pay (CAD)", "Profit (CAD)", 
                    "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
                ]
            ].rename(columns={
                "LS_FROM_ZONE": "From Zone",
                "LS_TO_ZONE": "To Zone",
                "LS_FREIGHT": "BILL_NUMBER",
                "CALLNAME": "Customer",
                "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
                "LS_LEG_DIST": "Leg Distance (miles)",
                "Bill Distance (miles)": "Bill Distance (miles)"
            })
    
            # Identify if each power unit is an Owner Operator
            owner_ops_units = set(owner_ops_fuel_df['VEHICLE_NO'])  # Get unique power units in the Owner Ops report
            merged_df['Is Owner Operator'] = merged_df['LS_POWER_UNIT'].isin(owner_ops_units)  # True if in Owner Ops
            
            # Set lease cost for Owner Ops
            lease_cost = 3100  # Fixed lease cost in CAD
            
            # Calculate Fuel Cost
            fuel_cost_multiplier = 1.45  # Multiplier for fuel cost calculation
            merged_df['Fuel Cost'] = merged_df['FUEL_QUANTITY_L'] * fuel_cost_multiplier
            
            # Calculate Fuel Cost for the selected power unit
            grand_fuel_cost = (
                filtered_view['FUEL_QUANTITY_L'].iloc[0] * fuel_cost_multiplier
                if 'FUEL_QUANTITY_L' in filtered_view and not filtered_view.empty else 0
            )
            
            # Add Lease Cost column to the route summary
            route_summary_df['Lease Cost'] = ""  # Initialize as blank for all rows
            
            # Add the Grand Totals row
            if filtered_view['LS_POWER_UNIT'].iloc[0] in owner_ops_units:
                # Owner Ops: Apply lease cost in the grand total row
                total_lease_cost = lease_cost  # Lease cost is $3100
                grand_totals = pd.DataFrame([{
                    "Route": "Grand Totals",
                    "From Zone": "",
                    "To Zone": "",
                    "BILL_NUMBER": "",
                    "Customer": "",
                    "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
                    "Leg Distance (miles)": route_summary_df["Leg Distance (miles)"].sum(),
                    "Bill Distance (miles)": route_summary_df["Bill Distance (miles)"].sum(),
                    "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Bill Distance (miles)"].sum()
                    if route_summary_df["Bill Distance (miles)"].sum() != 0 else 0,
                    "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
                    "Lease Cost": f"${lease_cost:,.2f}",  # Display $3100 in Grand Total row
                    "Fuel Cost": f"${grand_fuel_cost:,.2f}",
                    "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() -
                                    route_summary_df["Driver Pay (CAD)"].sum() -
                                    lease_cost -  # Include lease cost in profit calculation
                                    grand_fuel_cost,
                    "LS_ACTUAL_DATE": "",
                    "LS_LEG_NOTE": "",
                    "Highlight": None,
                    "LS_POWER_UNIT": ""
                }])
            else:
                # Company Trucks: No lease cost in the grand total row
                grand_totals = pd.DataFrame([{
                    "Route": "Grand Totals",
                    "From Zone": "",
                    "To Zone": "",
                    "BILL_NUMBER": "",
                    "Customer": "",
                    "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
                    "Leg Distance (miles)": route_summary_df["Leg Distance (miles)"].sum(),
                    "Bill Distance (miles)": route_summary_df["Bill Distance (miles)"].sum(),
                    "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Bill Distance (miles)"].sum()
                    if route_summary_df["Bill Distance (miles)"].sum() != 0 else 0,
                    "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
                    "Lease Cost": "",  # Blank for Company Trucks
                    "Fuel Cost": f"${grand_fuel_cost:,.2f}",
                    "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() -
                                    route_summary_df["Driver Pay (CAD)"].sum() -
                                    grand_fuel_cost,  # No lease cost deduction
                    "LS_ACTUAL_DATE": "",
                    "LS_LEG_NOTE": "",
                    "Highlight": None,
                    "LS_POWER_UNIT": ""
                }])
            
            # Append the Grand Total row
            route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
            
            # Rearrange columns to position Lease Cost and Fuel Cost appropriately
            route_summary_df = route_summary_df[
                [
                    "Route", "From Zone", "To Zone", "BILL_NUMBER", "Customer", "Total Charge (CAD)",
                    "Leg Distance (miles)", "Bill Distance (miles)", "Revenue per Mile", "Driver Pay (CAD)",
                    "Lease Cost", "Fuel Cost", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
                ]
            ]
            
            # Format numeric columns for display
            for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)", "Fuel Cost"]:
                route_summary_df[col] = route_summary_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
                )
            
            # Rearrange columns so Lease Cost and Fuel Cost appear after Driver Pay
            route_summary_df = route_summary_df[
                [
                    "Route", "From Zone", "To Zone", "BILL_NUMBER", "Customer", "Total Charge (CAD)",
                    "Leg Distance (miles)", "Bill Distance (miles)", "Revenue per Mile", "Driver Pay (CAD)",
                    "Lease Cost", "Fuel Cost", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
                ]
            ]
            
            # Format currency and numeric columns
            for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)", "Lease Cost", "Fuel Cost"]:
                route_summary_df[col] = route_summary_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
                )
            
            # Format distance columns to 1 decimal place
            for col in ["Leg Distance (miles)", "Bill Distance (miles)"]:
                route_summary_df[col] = route_summary_df[col].apply(
                    lambda x: f"{x:,.1f}" if pd.notna(x) and isinstance(x, (float, int)) else x
                )
            
            # Define row styling
            def highlight_rows(row):
                if row['Route'] == "Grand Totals":
                    return ['background-color: #f7c8c8'] * len(row)
                elif row['Highlight'] == 1:
                    return ['background-color: #c8e0f7'] * len(row)
                else:
                    return ['background-color: #f7f7c8'] * len(row)
            
            # Apply styling and display the table
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

if uploaded_legsum_file and uploaded_tlorder_driverpay_file and uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
    with st.expander("All Grand Totals"):

        # Check if each power unit is in the Owner Ops report
        owner_ops_units = set(owner_ops_fuel_df['VEHICLE_NO'])  # Get unique power units in the Owner Ops report
        merged_df['Is Owner Operator'] = merged_df['LS_POWER_UNIT'].isin(owner_ops_units)  # True if in Owner Ops

        # Set lease cost for Owner Ops
        lease_cost = 3100  # Fixed lease cost in CAD

        # Calculate Fuel Cost per unit (already aggregated in the data)
        fuel_cost_multiplier = 1.45  # Multiplier for fuel cost calculation
        fuel_cost_per_unit = isaac_combined_fuel_df.groupby('VEHICLE_NO').agg({'FUEL_QUANTITY_L': 'sum'}).reset_index()
        fuel_cost_per_unit['Fuel Cost'] = fuel_cost_per_unit['FUEL_QUANTITY_L'] * fuel_cost_multiplier
        fuel_cost_per_unit = fuel_cost_per_unit.rename(columns={'VEHICLE_NO': 'LS_POWER_UNIT'})[['LS_POWER_UNIT', 'Fuel Cost']]

        # Filter valid rows based on Route Summary logic
        valid_rows = merged_df[
            (merged_df['TOTAL_CHARGE_CAD'].notna()) &  # Non-NaN Total Charge
            (merged_df['LS_POWER_UNIT'].notna()) &  # Valid Power Unit
            (merged_df['Bill Distance (miles)'].notna()) &  # Non-NaN Bill Distance
            (merged_df['BILL_NUMBER'].notna())  # Ensure BILL_NUMBER exists
        ].copy()

        # Aggregate at the BILL_NUMBER level first to avoid duplicate summing
        bill_aggregates = valid_rows.groupby(['LS_POWER_UNIT', 'BILL_NUMBER']).agg({
            'TOTAL_CHARGE_CAD': 'first',  # Take the unique charge per BILL_NUMBER
            'Bill Distance (miles)': 'first',  # Take the unique distance per BILL_NUMBER
            'TOTAL_PAY_SUM': 'first'  # Take the unique driver pay per BILL_NUMBER
        }).reset_index()

        # Summing aggregated values across each power unit
        power_unit_aggregates = bill_aggregates.groupby('LS_POWER_UNIT').agg({
            'TOTAL_CHARGE_CAD': 'sum',  # Sum Total Charges for all BILL_NUMBERs per power unit
            'Bill Distance (miles)': 'sum',  # Sum distances for all BILL_NUMBERs per power unit
            'TOTAL_PAY_SUM': 'sum'  # Sum driver pay for all BILL_NUMBERs per power unit
        }).reset_index()

        # Merge with fuel cost per unit (fuel cost already summed per unit)
        all_grand_totals = power_unit_aggregates.merge(
            fuel_cost_per_unit, on='LS_POWER_UNIT', how='left'
        ).fillna({'Fuel Cost': 0})  # Ensure Fuel Cost is 0 for missing units

        # Add calculated fields
        all_grand_totals['Type'] = all_grand_totals['LS_POWER_UNIT'].apply(
            lambda unit: "Owner Ops" if unit in owner_ops_units else "Company Truck"
        )
        all_grand_totals['Lease Cost'] = all_grand_totals['LS_POWER_UNIT'].apply(
            lambda unit: lease_cost if unit in owner_ops_units else 0
        )
        all_grand_totals['Revenue per Mile'] = all_grand_totals.apply(
            lambda row: row['TOTAL_CHARGE_CAD'] / row['Bill Distance (miles)']
            if row['Bill Distance (miles)'] > 0 else 0,
            axis=1
        )
        all_grand_totals['Profit (CAD)'] = (
            all_grand_totals['TOTAL_CHARGE_CAD']
            - all_grand_totals['TOTAL_PAY_SUM']
            - all_grand_totals['Lease Cost']
            - all_grand_totals['Fuel Cost']
        )

        # Format the table for display
        all_grand_totals_display = all_grand_totals[[
            'LS_POWER_UNIT', 'Type', 'TOTAL_CHARGE_CAD', 'Bill Distance (miles)',
            'Revenue per Mile', 'TOTAL_PAY_SUM', 'Lease Cost', 'Fuel Cost', 'Profit (CAD)'
        ]].rename(columns={
            'LS_POWER_UNIT': 'Power Unit',
            'TOTAL_CHARGE_CAD': 'Total Charge (CAD)',
            'Bill Distance (miles)': 'Bill Distance (miles)',
            'TOTAL_PAY_SUM': 'Driver Pay (CAD)'
        })

        # Format numeric columns for display
        all_grand_totals_display['Bill Distance (miles)'] = all_grand_totals_display['Bill Distance (miles)'].apply(
            lambda x: f"{x:,.1f}" if pd.notna(x) and isinstance(x, (float, int)) else x
        )
        for col in ['Total Charge (CAD)', 'Revenue per Mile', 'Driver Pay (CAD)', 'Lease Cost', 'Fuel Cost', 'Profit (CAD)']:
            all_grand_totals_display[col] = all_grand_totals_display[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )

        # Display the table
        st.write("This table contains grand totals for all power units:")
        st.dataframe(all_grand_totals_display, use_container_width=True)

else:
    st.warning("Please upload all CSV and XLSX files to proceed.")
    
