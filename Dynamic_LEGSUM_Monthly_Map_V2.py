### adding lease and fuel costs, grand total table 

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

def load_and_preprocess_data(uploaded_legsum_file, uploaded_tlorder_driverpay_file, 
                             uploaded_isaac_owner_ops_file, uploaded_isaac_company_trucks_file):
    """
    Load and preprocess data files, including LEGSUM, TLORDER+DRIVERPAY, and ISAAC Fuel Reports.
    """
    # Load city coordinates
    city_coordinates_df = load_city_coordinates()
    
    # Preprocess LEGSUM and TLORDER+DRIVERPAY
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)
    
    # Combine ISAAC Fuel Reports if both are uploaded
    isaac_combined_fuel_df = None
    if uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
        isaac_combined_fuel_df = pd.concat(
            [
                preprocess_new_isaac_fuel(uploaded_isaac_owner_ops_file),
                preprocess_new_isaac_fuel(uploaded_isaac_company_trucks_file)
            ],
            ignore_index=True
        )

    # Merge data into a single DataFrame
    merged_df = legsum_df.merge(
        tlorder_driverpay_df,
        left_on='LS_FREIGHT',
        right_on='BILL_NUMBER',
        how='left'
    )
    
    if isaac_combined_fuel_df is not None:
        merged_df = merged_df.merge(
            isaac_combined_fuel_df,
            left_on='LS_POWER_UNIT',
            right_on='VEHICLE_NO',
            how='left'
        )
        merged_df.drop(columns=['VEHICLE_NO'], inplace=True, errors='ignore')

    # Add province columns (if applicable)
    if 'ORIGPROV' in merged_df and 'DESTPROV' in merged_df:
        merged_df['ORIGPROV'] = merged_df['ORIGPROV']
        merged_df['DESTPROV'] = merged_df['DESTPROV']
    
    return merged_df


def display_power_unit_finder(merged_df):
    """
    Display the 'Power Unit Finder' section of the app.
    """
    with st.expander("Power Unit Finder", expanded=True):  # Collapsible section
        st.header("Power Unit Finder")

        # Dropdown for Customer (CALLNAME)
        callname_options = sorted(merged_df['CALLNAME'].dropna().unique())
        selected_callname = st.selectbox("Select Customer (CALLNAME):", options=callname_options)

        # Filter data for selected customer
        filtered_data = merged_df[merged_df['CALLNAME'] == selected_callname]

        # Dropdowns for Origin and Destination Provinces
        origprov_options = ["All"] + sorted(filtered_data['ORIGPROV'].dropna().unique())
        destprov_options = ["All"] + sorted(filtered_data['DESTPROV'].dropna().unique())

        selected_origprov = st.selectbox("Select Origin Province (ORIGPROV):", options=origprov_options)
        selected_destprov = st.selectbox("Select Destination Province (DESTPROV):", options=destprov_options)

        # Add Date Range Filtering
        st.write("### Select Date Range:")
        if not filtered_data.empty:
            min_date = filtered_data['LS_ACTUAL_DATE'].min().date()
            max_date = filtered_data['LS_ACTUAL_DATE'].max().date()

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
            valid_bills = filtered_data[
                ((filtered_data['ORIGPROV'] == selected_origprov) | (selected_origprov == "All")) &
                ((filtered_data['DESTPROV'] == selected_destprov) | (selected_destprov == "All"))
            ]['BILL_NUMBER'].unique()

            # Filter original dataset to include all legs for these BILL_NUMBERs
            filtered_data = filtered_data[filtered_data['BILL_NUMBER'].isin(valid_bills)]

        if filtered_data.empty:
            st.warning("No results found for the selected criteria.")
        else:
            grouped = filtered_data.groupby('BILL_NUMBER')
            for bill_number, group in grouped:
                # Get overall ORIGPROV -> DESTPROV for the BILL_NUMBER
                bill_origprov = group['ORIGPROV'].iloc[0]
                bill_destprov = group['DESTPROV'].iloc[0]

                # Display the BILL_NUMBER and its ORIGPROV -> DESTPROV
                st.write(f"### Bill Number: {bill_number} ({bill_origprov} to {bill_destprov})")

                # Sort and display the data
                group = group.sort_values(by=['LS_ACTUAL_DATE', 'LS_LEG_SEQ'])
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
                st.write(bill_table)


# Main logic
if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    merged_df = load_and_preprocess_data(uploaded_legsum_file, uploaded_tlorder_driverpay_file, 
                                         uploaded_isaac_owner_ops_file, uploaded_isaac_company_trucks_file)
    display_power_unit_finder(merged_df)


def preprocess_merged_data(uploaded_legsum_file, uploaded_tlorder_driverpay_file, 
                           uploaded_isaac_owner_ops_file=None, uploaded_isaac_company_trucks_file=None):
    """
    Preprocess and merge data from uploaded files, including LEGSUM, TLORDER+DRIVERPAY, and ISAAC Fuel Reports.
    
    Args:
        uploaded_legsum_file (UploadedFile): LEGSUM CSV file.
        uploaded_tlorder_driverpay_file (UploadedFile): Pre-merged TLORDER + DRIVERPAY CSV file.
        uploaded_isaac_owner_ops_file (UploadedFile, optional): ISAAC Owner Ops Fuel Report Excel file.
        uploaded_isaac_company_trucks_file (UploadedFile, optional): ISAAC Company Trucks Fuel Report Excel file.

    Returns:
        pd.DataFrame: Merged and preprocessed DataFrame.
    """
    # Load city coordinates
    city_coordinates_df = load_city_coordinates()

    # Preprocess LEGSUM and TLORDER+DRIVERPAY
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Preprocess ISAAC fuel reports if provided
    if uploaded_isaac_owner_ops_file and uploaded_isaac_company_trucks_file:
        owner_ops_fuel_df = preprocess_new_isaac_fuel(uploaded_isaac_owner_ops_file)
        company_trucks_fuel_df = preprocess_new_isaac_fuel(uploaded_isaac_company_trucks_file)
        isaac_combined_fuel_df = pd.concat([owner_ops_fuel_df, company_trucks_fuel_df], ignore_index=True)
    else:
        isaac_combined_fuel_df = None

    # Merge LEGSUM and TLORDER+DRIVERPAY
    merged_df = legsum_df.merge(
        tlorder_driverpay_df,
        left_on='LS_FREIGHT',
        right_on='BILL_NUMBER',
        how='left'
    )

    # Merge with ISAAC fuel data if available
    if isaac_combined_fuel_df is not None:
        merged_df = merged_df.merge(
            isaac_combined_fuel_df,
            left_on='LS_POWER_UNIT',
            right_on='VEHICLE_NO',
            how='left'
        )
        # Drop VEHICLE_NO column after merging
        merged_df.drop(columns=['VEHICLE_NO'], inplace=True, errors='ignore')

    return merged_df

def add_calculated_fields(merged_df, exchange_rate=1.38):
    """
    Add calculated fields such as total charges in CAD, revenue per mile, profit, and clean up distances.
    
    Args:
        merged_df (pd.DataFrame): Merged DataFrame.
        exchange_rate (float): Exchange rate for USD to CAD conversion (default is 1.38).
    
    Returns:
        pd.DataFrame: DataFrame with added calculated fields.
    """
    # Ensure CHARGES and XCHARGES are numeric
    merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
    merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')

    # Calculate TOTAL_CHARGE_CAD based on currency
    merged_df['TOTAL_CHARGE_CAD'] = np.where(
        merged_df['CURRENCY_CODE'] == 'USD',  # If USD, convert to CAD
        (merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)) * exchange_rate,
        merged_df['CHARGES'].fillna(0) + merged_df['XCHARGES'].fillna(0)  # Otherwise, leave as-is
    )

    # Ensure LS_LEG_DIST and DISTANCE are numeric
    merged_df['LS_LEG_DIST'] = pd.to_numeric(merged_df['LS_LEG_DIST'], errors='coerce')
    merged_df['DISTANCE'] = pd.to_numeric(merged_df['DISTANCE'], errors='coerce')

    # Convert distances from KM to miles if applicable
    merged_df['Bill Distance (miles)'] = np.where(
        merged_df['DISTANCE_UNITS'] == 'KM',
        merged_df['DISTANCE'] * 0.62,  # Convert KM to miles
        merged_df['DISTANCE']  # Otherwise, keep as-is
    )

    # Set LS_LEG_DIST to NaN for invalid values
    merged_df['LS_LEG_DIST'] = np.where(merged_df['LS_LEG_DIST'] > 0, merged_df['LS_LEG_DIST'], np.nan)

    # Calculate Revenue per Mile
    merged_df['Revenue per Mile'] = np.where(
        (merged_df['TOTAL_CHARGE_CAD'].notna()) & 
        (merged_df['Bill Distance (miles)'].notna()) & 
        (merged_df['Bill Distance (miles)'] > 0),
        merged_df['TOTAL_CHARGE_CAD'] / merged_df['Bill Distance (miles)'],
        np.nan  # Set to NaN if any value is missing or invalid
    )

    # Assign Driver Pay only for valid BILL_NUMBERs
    merged_df['Driver Pay (CAD)'] = np.where(
        merged_df['LS_FREIGHT'].notna(),  # Only assign pay if BILL_NUMBER exists
        merged_df['TOTAL_PAY_SUM'].fillna(0),
        0
    )

    # Calculate Profit (CAD)
    merged_df['Profit (CAD)'] = np.where(
        merged_df['TOTAL_CHARGE_CAD'].notna(),
        merged_df['TOTAL_CHARGE_CAD'] - merged_df['Driver Pay (CAD)'],
        np.nan  # Leave blank if charges are missing
    )

    return merged_df

def extract_month_year_title(merged_df):
    """
    Extract the month and year from the dataset for display purposes.

    Args:
        merged_df (pd.DataFrame): DataFrame with a date column 'LS_ACTUAL_DATE'.

    Returns:
        str: Title indicating the month and year, or "Unknown Month" if no valid dates exist.
    """
    merged_df['LS_ACTUAL_DATE'] = pd.to_datetime(merged_df['LS_ACTUAL_DATE'], errors='coerce')
    
    if not merged_df['LS_ACTUAL_DATE'].isna().all():
        # Extract month name and year
        month_name = merged_df['LS_ACTUAL_DATE'].dt.month_name().iloc[0]
        year = merged_df['LS_ACTUAL_DATE'].dt.year.iloc[0]
        return f"{month_name} {year}"
    return "Unknown Month"

def handle_missing_locations(merged_df):
    """
    Identify missing locations and calculate straight-line distances for valid rows.

    Args:
        merged_df (pd.DataFrame): DataFrame with coordinates and location columns.

    Returns:
        tuple: DataFrame with updated distances, DataFrame of missing locations.
    """
    # Identify missing origins and destinations
    missing_origins = merged_df[
        pd.isna(merged_df['LEGO_LAT']) | pd.isna(merged_df['LEGO_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = merged_df[
        pd.isna(merged_df['LEGD_LAT']) | pd.isna(merged_df['LEGD_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    # Combine missing locations
    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Fill Straight Distance for valid rows
    merged_df['Straight Distance'] = np.where(
        pd.isna(merged_df['LEGO_LAT']) | pd.isna(merged_df['LEGD_LAT']),
        np.nan,
        calculate_haversine(merged_df)
    )

    return merged_df, missing_locations

def filter_by_power_unit_and_driver(merged_df, selected_punit, selected_driver, start_date, end_date):
    """
    Filter the merged DataFrame by selected power unit, driver, and date range.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame.
        selected_punit (str): Selected power unit.
        selected_driver (str): Selected driver ID.
        start_date (date): Start date for filtering.
        end_date (date): End date for filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Filter by power unit
    filtered_view = merged_df[merged_df['LS_POWER_UNIT'] == selected_punit].copy()

    # Filter by driver if specified
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['LS_DRIVER'] == selected_driver]

    # Filter by date range
    filtered_view = filtered_view[
        (filtered_view['LS_ACTUAL_DATE'].dt.date >= start_date) &
        (filtered_view['LS_ACTUAL_DATE'].dt.date <= end_date)
    ].copy()

    # Deduplicate rows based on key columns
    filtered_view['Route'] = filtered_view['LEGO_ZONE_DESC'] + " to " + filtered_view['LEGD_ZONE_DESC']
    filtered_view = filtered_view.drop_duplicates(subset=['LS_POWER_UNIT', 'Route', 'LS_ACTUAL_DATE'], keep='first')

    return filtered_view

# Process merged data
merged_df = preprocess_merged_data(
    uploaded_legsum_file, 
    uploaded_tlorder_driverpay_file, 
    uploaded_isaac_owner_ops_file, 
    uploaded_isaac_company_trucks_file
)

# Add calculated fields
merged_df = add_calculated_fields(merged_df)

# Extract month-year title
month_year_title = extract_month_year_title(merged_df)
st.header(f"Table and Map for Power Unit - {month_year_title}")

# Handle missing locations
merged_df, missing_locations = handle_missing_locations(merged_df)

# Get power unit and driver options
punit_options = sorted(merged_df['LS_POWER_UNIT'].unique())
selected_punit = st.selectbox("Select Power Unit:", options=punit_options)

relevant_drivers = merged_df['LS_DRIVER'].dropna().unique()
driver_options = ["All"] + sorted(relevant_drivers.astype(str))
selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)

# Date range filtering
min_date = merged_df['LS_ACTUAL_DATE'].min().date()
max_date = merged_df['LS_ACTUAL_DATE'].max().date()
start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Filter by power unit and driver
filtered_view = filter_by_power_unit_and_driver(merged_df, selected_punit, selected_driver, start_date, end_date)

def generate_route_summary(filtered_view, owner_ops_units, lease_cost=3100, fuel_cost_multiplier=1.45):
    # Add Lease Cost and Fuel Cost columns
    filtered_view['Lease Cost'] = ""
    filtered_view['Fuel Cost'] = filtered_view['FUEL_QUANTITY_L'] * fuel_cost_multiplier

    # Add Profit calculation
    filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - \
        filtered_view['Driver Pay (CAD)'] - \
        filtered_view['Fuel Cost']

    # Check if the selected power unit is an Owner Operator
    is_owner_op = filtered_view['LS_POWER_UNIT'].iloc[0] in owner_ops_units

    # Add Grand Totals row
    grand_totals = pd.DataFrame([{
        "Route": "Grand Totals",
        "From Zone": "",
        "To Zone": "",
        "BILL_NUMBER": "",
        "Customer": "",
        "Total Charge (CAD)": filtered_view["TOTAL_CHARGE_CAD"].sum(),
        "Leg Distance (miles)": filtered_view["LS_LEG_DIST"].sum(),
        "Bill Distance (miles)": filtered_view["Bill Distance (miles)"].sum(),
        "Revenue per Mile": filtered_view["TOTAL_CHARGE_CAD"].sum() / filtered_view["Bill Distance (miles)"].sum()
        if filtered_view["Bill Distance (miles)"].sum() != 0 else 0,
        "Driver Pay (CAD)": filtered_view["Driver Pay (CAD)"].sum(),
        "Lease Cost": f"${lease_cost:,.2f}" if is_owner_op else "",
        "Fuel Cost": f"${filtered_view['Fuel Cost'].sum():,.2f}",
        "Profit (CAD)": filtered_view["TOTAL_CHARGE_CAD"].sum() -
                        filtered_view["Driver Pay (CAD)"].sum() -
                        (lease_cost if is_owner_op else 0) -
                        filtered_view["Fuel Cost"].sum(),
        "LS_ACTUAL_DATE": "",
        "LS_LEG_NOTE": "",
        "Highlight": None,
        "LS_POWER_UNIT": ""
    }])

    # Concatenate Grand Totals row with the route summary
    route_summary_df = pd.concat([filtered_view, grand_totals], ignore_index=True)

    # Rearrange columns for better display
    route_summary_df = route_summary_df[
        [
            "Route", "From Zone", "To Zone", "BILL_NUMBER", "Customer", "Total Charge (CAD)",
            "Leg Distance (miles)", "Bill Distance (miles)", "Revenue per Mile", "Driver Pay (CAD)",
            "Lease Cost", "Fuel Cost", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight", "LS_POWER_UNIT"
        ]
    ]

    # Format currency columns for display
    for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)", "Lease Cost", "Fuel Cost"]:
        route_summary_df[col] = route_summary_df[col].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
        )

    # Format distance columns
    for col in ["Leg Distance (miles)", "Bill Distance (miles)"]:
        route_summary_df[col] = route_summary_df[col].apply(
            lambda x: f"{x:,.1f}" if pd.notna(x) and isinstance(x, (float, int)) else x
        )

    return route_summary_df

def generate_map_visualization(filtered_view, location_aggregates, missing_locations):
    fig = go.Figure()

    # Track city appearance sequence for labels
    location_sequence = {}
    label_counter = 1

    for _, row in filtered_view.iterrows():
        if row['LEGO_ZONE_DESC'] not in location_sequence:
            location_sequence[row['LEGO_ZONE_DESC']] = label_counter
            label_counter += 1
        if row['LEGD_ZONE_DESC'] not in location_sequence:
            location_sequence[row['LEGD_ZONE_DESC']] = label_counter
            label_counter += 1

        # Add line for the route
        fig.add_trace(go.Scattergeo(
            lon=[row['LEGO_LON'], row['LEGD_LON']],
            lat=[row['LEGO_LAT'], row['LEGD_LAT']],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Route"
        ))

        # Add markers for origin and destination
        fig.add_trace(go.Scattergeo(
            lon=[row['LEGO_LON']],
            lat=[row['LEGO_LAT']],
            mode="markers+text",
            text=f"{location_sequence[row['LEGO_ZONE_DESC']]}",
            marker=dict(size=8, color="blue"),
            name="Origin"
        ))

        fig.add_trace(go.Scattergeo(
            lon=[row['LEGD_LON']],
            lat=[row['LEGD_LAT']],
            mode="markers+text",
            text=f"{location_sequence[row['LEGD_ZONE_DESC']]}",
            marker=dict(size=8, color="red"),
            name="Destination"
        ))

    # Configure map layout
    fig.update_layout(
        title="Route Map Visualization",
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

    return fig

# Generate Route Summary
if not filtered_view.empty:
    route_summary_df = generate_route_summary(filtered_view, owner_ops_units)

    # Display the Route Summary
    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)

    # Handle Missing Locations
    missing_locations = merged_df[merged_df['LEGO_ZONE_DESC'].isna() | merged_df['LEGD_ZONE_DESC'].isna()]
    if not missing_locations.empty:
        st.write("Missing Location Coordinates:")
        st.dataframe(missing_locations, use_container_width=True)

    # Generate Map
    map_fig = generate_map_visualization(filtered_view, location_aggregates, missing_locations)
    st.plotly_chart(map_fig)
