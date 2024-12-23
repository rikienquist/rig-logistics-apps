import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer")

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

    def clean_location_name(name):
        return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().upper()

    city_coords['LOCATION'] = city_coords['LOCATION'].apply(clean_location_name)

    df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].apply(clean_location_name)
    df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].apply(clean_location_name)

    # Ensure no duplicates in city coordinates
    city_coords = city_coords.drop_duplicates(subset=['LOCATION'])

    # Merge for LEGO_ZONE_DESC (origin)
    lego_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(lego_coords, on="LEGO_ZONE_DESC", how="left")

    # Merge for LEGD_ZONE_DESC (destination)
    legd_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    df = df.merge(legd_coords, on="LEGD_ZONE_DESC", how="left")

    return df

@st.cache_data
def preprocess_tlorder(file):
    df = pd.read_csv(file, low_memory=False)
    df['CHARGES'] = pd.to_numeric(df['CHARGES'], errors='coerce')
    df['XCHARGES'] = pd.to_numeric(df['XCHARGES'], errors='coerce')
    df['TOTAL_CHARGE_CAD'] = df['CHARGES'] + df['XCHARGES']
    return df[['BILL_NUMBER', 'TOTAL_CHARGE_CAD']]

@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_AMT'] = pd.to_numeric(df['TOTAL_PAY_AMT'], errors='coerce')
    driver_pay_agg = df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum'}).reset_index()
    return driver_pay_agg

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    
    # Preprocess TLORDER data
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file)
    tlorder_df.rename(columns={'BILL_NUMBER': 'BILL_NUMBER_TLORDER'}, inplace=True)
    
    # Preprocess DRIVERPAY data
    driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)
    driver_pay_agg.rename(columns={'BILL_NUMBER': 'BILL_NUMBER_DRIVERPAY'}, inplace=True)

    # Merge TLORDER with DRIVERPAY on BILL_NUMBER
    merged_df = tlorder_df.merge(driver_pay_agg, left_on='BILL_NUMBER_TLORDER', right_on='BILL_NUMBER_DRIVERPAY', how='left')

    # Filter for valid routes, keeping rows with missing coordinates for later processing
    valid_routes = merged_df[
        (merged_df['ORIGCITY'] != merged_df['DESTCITY']) & 
        pd.notna(merged_df['DISTANCE'])
    ].copy()

    # Convert charges to numeric and calculate total charge in CAD
    exchange_rate = 1.38
    valid_routes['CHARGES'] = pd.to_numeric(valid_routes['CHARGES'], errors='coerce')
    valid_routes['XCHARGES'] = pd.to_numeric(valid_routes['XCHARGES'], errors='coerce')
    valid_routes['TOTAL_CHARGE_CAD'] = np.where(
        valid_routes['CURRENCY_CODE'] == 'USD',
        (valid_routes['CHARGES'] + valid_routes['XCHARGES']) * exchange_rate,
        valid_routes['CHARGES'] + valid_routes['XCHARGES']
    )

    # Filter for routes with non-zero charges
    filtered_df = valid_routes[valid_routes['TOTAL_CHARGE_CAD'] != 0].copy()

    # Fill missing PICK_UP_PUNIT values and calculate additional metrics
    filtered_df['PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].fillna("Unknown").astype(str)
    filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
    filtered_df['Profit (CAD)'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_AMT']

    # Handle effective date and create a "Month" column
    cutoff_date = pd.Timestamp("2024-10-01")
    filtered_df['Effective_Date'] = pd.to_datetime(
        np.where(
            pd.to_datetime(filtered_df['DELIVER_BY'], errors='coerce') >= cutoff_date,
            filtered_df['DELIVER_BY'],
            filtered_df['PICK_UP_BY']
        ),
        errors='coerce'
    )
    filtered_df['Month'] = filtered_df['Effective_Date'].dt.to_period('M')

    # Identify missing city coordinates for origin and destination
    missing_origins = filtered_df[
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['ORIG_LON'])
    ][['ORIGCITY', 'ORIGPROV']].drop_duplicates().rename(columns={
        'ORIGCITY': 'City', 'ORIGPROV': 'Province'
    })

    missing_destinations = filtered_df[
        pd.isna(filtered_df['DEST_LAT']) | pd.isna(filtered_df['DEST_LON'])
    ][['DESTCITY', 'DESTPROV']].drop_duplicates().rename(columns={
        'DESTCITY': 'City', 'DESTPROV': 'Province'
    })

    missing_cities = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Calculate Straight Distance using haversine formula
    filtered_df['Straight Distance'] = np.where(
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['DEST_LAT']),
        np.nan,
        calculate_haversine(filtered_df)
    )

    punit_options = sorted(filtered_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit (LS_POWER_UNIT):", options=punit_options)
    
    # Filter data by selected power unit
    filtered_view = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit].copy()
    
    # Extract relevant drivers for the selected power unit
    relevant_drivers = filtered_view['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    
    # Filter by selected driver if specified
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()
    
    # Extract unique months for the filtered view
    months = sorted(filtered_view['Month'].unique())
    if len(months) == 0:
        st.warning("No data available for the selected Power Unit and Driver ID.")
    else:
        # Handle selected month with session state
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
    
        # Filter data for the selected month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()
    
    if not month_data.empty:
        # Assign alternating colors for rows by day
        month_data = month_data.sort_values(by='Effective_Date')
        month_data['Day_Group'] = month_data['Effective_Date'].dt.date
        unique_days = list(month_data['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        month_data['Highlight'] = month_data['Day_Group'].map(day_colors)
    
        # Create the route summary DataFrame
        month_data['Profit (CAD)'] = month_data['TOTAL_CHARGE_CAD'] - month_data['TOTAL_PAY_AMT']
        month_data['Route'] = month_data['LEGO_ZONE_DESC'] + " to " + month_data['LEGD_ZONE_DESC']
    
        route_summary_df = month_data[[
            "Route", "BILL_NUMBER_TLORDER", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED",
            "TOTAL_CHARGE_CAD", "DISTANCE", "Straight Distance", "Revenue per Mile", 
            "DRIVER_ID", "TOTAL_PAY_AMT", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight"
        ]].rename(columns={
            "BILL_NUMBER_TLORDER": "BILL_NUMBER",
            "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
            "DISTANCE": "Distance (miles)",
            "Straight Distance": "Straight Distance (miles)",
            "TOTAL_PAY_AMT": "Driver Pay (CAD)"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "LS_TRIP_NUMBER": "",
            "LS_LEG_DIST": route_summary_df["LS_LEG_DIST"].sum(),
            "LS_MT_LOADED": "",
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
    
        # Combine route summary with grand totals
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
    
        # Format currency and numeric columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
            route_summary_df[col] = route_summary_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )
    
        # Display the summary table
        st.write("Route Summary:")
        st.dataframe(route_summary_df, use_container_width=True)
    
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
            month_data[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'DISTANCE', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGO_ZONE_DESC': 'Location'}
            ),
            month_data[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'DISTANCE', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGD_ZONE_DESC': 'Location'}
            )
        ], ignore_index=True)
        
        # Clean and aggregate the combined data
        location_aggregates = location_aggregates.groupby('Location', as_index=False).agg({
            'TOTAL_CHARGE_CAD': 'sum',
            'DISTANCE': 'sum',
            'TOTAL_PAY_AMT': 'sum'
        })
        
        location_aggregates['Revenue per Mile'] = location_aggregates['TOTAL_CHARGE_CAD'] / location_aggregates['DISTANCE']
        location_aggregates['Profit (CAD)'] = location_aggregates['TOTAL_CHARGE_CAD'] - location_aggregates['TOTAL_PAY_AMT'].fillna(0)
        
        # Function to fetch aggregate values for a location
        def get_location_aggregates(location):
            match = location_aggregates[location_aggregates['Location'] == location]
            if not match.empty:
                total_charge = match['TOTAL_CHARGE_CAD'].iloc[0]
                distance = match['DISTANCE'].iloc[0]
                driver_pay = match['TOTAL_PAY_AMT'].iloc[0]
                profit = match['Profit (CAD)'].iloc[0]
                rpm = match['Revenue per Mile'].iloc[0]
                return total_charge, distance, driver_pay, profit, rpm
            return 0, 0, 0, 0, 0
        
        # Generate the map
        fig = go.Figure()
        
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
                lon=[row['ORIG_LON']],
                lat=[row['ORIG_LAT']],
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
                lon=[row['DEST_LON']],
                lat=[row['DEST_LAT']],
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
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
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
            (pd.isna(month_data['ORIG_LAT']) | pd.isna(month_data['ORIG_LON']))
        ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={
            'LEGO_ZONE_DESC': 'Location'
        })
        
        relevant_missing_destinations = month_data[
            (pd.isna(month_data['DEST_LAT']) | pd.isna(month_data['DEST_LON']))
        ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={
            'LEGD_ZONE_DESC': 'Location'
        })
        
        relevant_missing_locations = pd.concat([relevant_missing_origins, relevant_missing_destinations]).drop_duplicates()
        
        if not relevant_missing_locations.empty:
            st.write("### Locations Missing Coordinates")
            st.dataframe(relevant_missing_locations, use_container_width=True)
        else:
            st.info("All relevant locations have valid coordinates.")

    else:
        st.warning("No data available for the selected Power Unit and Driver ID.")
else:
    st.warning("Please upload the required LEGSUM CSV file. Optional TLORDER and DRIVERPAY files can be added for enhanced analysis.")



