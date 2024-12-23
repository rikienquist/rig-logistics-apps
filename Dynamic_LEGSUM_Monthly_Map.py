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

    # Clean location names to match coordinates
    def clean_location_name(name):
        return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().upper()

    city_coords['LOCATION'] = city_coords['LOCATION'].apply(clean_location_name)
    df['LEGO_ZONE_DESC'] = df['LEGO_ZONE_DESC'].apply(clean_location_name)
    df['LEGD_ZONE_DESC'] = df['LEGD_ZONE_DESC'].apply(clean_location_name)

    # Ensure unique location coordinates
    city_coords = city_coords.drop_duplicates(subset=['LOCATION'])

    # Merge for origin and destination coordinates
    origin_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(origin_coords, on="LEGO_ZONE_DESC", how="left")

    dest_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    df = df.merge(dest_coords, on="LEGD_ZONE_DESC", how="left")

    return df

@st.cache_data
def preprocess_tlorder(file):
    df = pd.read_csv(file, low_memory=False)
    df.rename(columns={"BILL_NUMBER": "TLORDER_BILL_NUMBER"}, inplace=True)
    df['CHARGES'] = pd.to_numeric(df['CHARGES'], errors='coerce')
    df['XCHARGES'] = pd.to_numeric(df['XCHARGES'], errors='coerce')
    return df

@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df.rename(columns={"BILL_NUMBER": "DRIVERPAY_BILL_NUMBER"}, inplace=True)
    df['TOTAL_PAY_AMT'] = pd.to_numeric(df['TOTAL_PAY_AMT'], errors='coerce')
    return df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
st.write("LEGSUM Columns:", legsum_df.columns.tolist())
st.write("TLORDER Columns:", tlorder_df.columns.tolist())
st.write("DRIVERPAY Columns:", driverpay_df.columns.tolist())


if uploaded_legsum_file and uploaded_tlorder_file and uploaded_driverpay_file:
    # Load city coordinates
    city_coordinates_df = load_city_coordinates()

    # Preprocess LEGSUM
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)

    # Preprocess TLORDER
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file)

    # Preprocess DRIVERPAY
    driverpay_df = preprocess_driverpay(uploaded_driverpay_file)

    # Merge LEGSUM with TLORDER on LS_FREIGHT to TLORDER_BILL_NUMBER
    legsum_df = legsum_df.merge(tlorder_df, left_on='LS_FREIGHT', right_on='TLORDER_BILL_NUMBER', how='left')

    # Merge the resulting dataset with DRIVERPAY on LS_FREIGHT to DRIVERPAY_BILL_NUMBER
    legsum_df = legsum_df.merge(driverpay_df, left_on='LS_FREIGHT', right_on='DRIVERPAY_BILL_NUMBER', how='left')

    # Define exchange rate for USD to CAD
    exchange_rate = 1.38

    # Ensure CHARGES and XCHARGES are numeric, replacing invalid entries with NaN
    legsum_df['CHARGES'] = pd.to_numeric(legsum_df['CHARGES'], errors='coerce')
    legsum_df['XCHARGES'] = pd.to_numeric(legsum_df['XCHARGES'], errors='coerce')

    # Calculate TOTAL_CHARGE_CAD for rows with a BILL_NUMBER
    legsum_df['TOTAL_CHARGE_CAD'] = (legsum_df['CHARGES'].fillna(0) + legsum_df['XCHARGES'].fillna(0)) * exchange_rate

    # Ensure LS_LEG_DIST is numeric, and replace invalid or zero values with NaN
    legsum_df['LS_LEG_DIST'] = pd.to_numeric(legsum_df['LS_LEG_DIST'], errors='coerce')
    legsum_df['LS_LEG_DIST'] = np.where(legsum_df['LS_LEG_DIST'] > 0, legsum_df['LS_LEG_DIST'], np.nan)

    # Safely calculate Revenue per Mile (RPM)
    legsum_df['Revenue per Mile'] = np.where(
        pd.notna(legsum_df['TOTAL_CHARGE_CAD']) & pd.notna(legsum_df['LS_LEG_DIST']),
        legsum_df['TOTAL_CHARGE_CAD'] / legsum_df['LS_LEG_DIST'],
        np.nan
    )

    # Calculate Profit (CAD)
    legsum_df['Profit (CAD)'] = legsum_df['TOTAL_CHARGE_CAD'] - legsum_df['TOTAL_PAY_AMT'].fillna(0)

    # Extract month for grouping
    legsum_df['Month'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE']).dt.to_period('M')

    # Identify locations missing in the coordinates dataset
    missing_origins = legsum_df[
        pd.isna(legsum_df['ORIG_LAT']) | pd.isna(legsum_df['ORIG_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = legsum_df[
        pd.isna(legsum_df['DEST_LAT']) | pd.isna(legsum_df['DEST_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Fill missing Straight Distance as np.nan for rows missing coordinates
    legsum_df['Straight Distance'] = np.where(
        pd.isna(legsum_df['ORIG_LAT']) | pd.isna(legsum_df['DEST_LAT']),
        np.nan,
        calculate_haversine(legsum_df)
    )

    legsum_df['LS_POWER_UNIT'] = legsum_df['LS_POWER_UNIT'].astype(str)
    punit_options = sorted(legsum_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
    
    # Filter data by selected Power Unit
    filtered_view = legsum_df[legsum_df['LS_POWER_UNIT'] == selected_punit].copy()
    
    # Populate Driver ID options based on the filtered Power Unit
    relevant_drivers = filtered_view['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    
    # Further filter by selected Driver ID if not "All"
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()
    
    # Filter by available months
    months = sorted(filtered_view['Month'].unique())
    if len(months) == 0:
        st.warning("No data available for the selected Power Unit and Driver ID.")
    else:
        # Ensure the month is consistent across session states
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]
    
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
    
        # Filter data for the selected month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()
    
        if not month_data.empty:
            # Sort and group by day for alternating row highlights
            month_data = month_data.sort_values(by='LS_ACTUAL_DATE')
            month_data['Day_Group'] = month_data['LS_ACTUAL_DATE'].dt.date
            unique_days = list(month_data['Day_Group'].unique())
            day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
            month_data['Highlight'] = month_data['Day_Group'].map(day_colors)
    
            # Create Route Summary DataFrame
            route_summary_df = month_data.assign(
                Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
            )[[  # Columns for display
                "Route", "LS_FREIGHT", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Straight Distance",
                "Revenue per Mile", "DRIVER_ID", "TOTAL_PAY_AMT", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight"
            ]].rename(columns={
                "LS_FREIGHT": "BILL_NUMBER",
                "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
                "LS_LEG_DIST": "Distance (miles)",
                "Straight Distance": "Straight Distance (miles)",
                "TOTAL_PAY_AMT": "Driver Pay (CAD)"
            })
    
            # Calculate grand totals for the route summary
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
    
            # Aggregate data by location
            location_aggregates = pd.concat([
                month_data[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                    columns={'LEGO_ZONE_DESC': 'Location'}
                ),
                month_data[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                    columns={'LEGD_ZONE_DESC': 'Location'}
                )
            ], ignore_index=True)
    
            # Aggregate totals for each location
            location_aggregates = location_aggregates.groupby(['Location'], as_index=False).agg({
                'TOTAL_CHARGE_CAD': 'sum',
                'LS_LEG_DIST': 'sum',
                'TOTAL_PAY_AMT': 'sum'
            })
    
            location_aggregates['Revenue per Mile'] = location_aggregates['TOTAL_CHARGE_CAD'] / location_aggregates['LS_LEG_DIST']
            location_aggregates['Profit (CAD)'] = location_aggregates['TOTAL_CHARGE_CAD'] - location_aggregates['TOTAL_PAY_AMT']
    
            # Helper function to fetch location aggregates
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


            fig = go.Figure()

            # Track sequence of location appearance for labeling
            location_sequence = {location: [] for location in set(month_data['LEGO_ZONE_DESC']).union(month_data['LEGD_ZONE_DESC'])}
            label_counter = 1
            for _, row in month_data.iterrows():
                location_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
                label_counter += 1
                location_sequence[row['LEGD_ZONE_DESC']].append(label_counter)
                label_counter += 1
            
            # Initialize legend tracking
            legend_added = {"Origin": False, "Destination": False, "Route": False}
            
            # Add map elements
            for _, row in month_data.iterrows():
                origin_sequence = ", ".join(map(str, location_sequence[row['LEGO_ZONE_DESC']]))
                destination_sequence = ", ".join(map(str, location_sequence[row['LEGD_ZONE_DESC']]))
            
                # Aggregated values for origin location
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
            
                # Aggregated values for destination location
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
            
            # Configure map layout
            fig.update_layout(
                title=f"Routes for {selected_month} - Power Unit: {selected_punit}, Driver ID: {selected_driver}",
                geo=dict(scope="north america", projection_type="mercator"),
            )
            
            # Display map
            st.plotly_chart(fig)
            
            # Identify and display missing locations
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
