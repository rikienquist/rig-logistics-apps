import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize date range for filtering
st.title("LEGSUM Trip Map Viewer")
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# File upload section
uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file", type="csv")
uploaded_driverpay_file = st.file_uploader("Upload DRIVERPAY CSV file", type="csv")

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
    df = df[(df['LS_ACTUAL_DATE'] >= pd.Timestamp(start_date)) & (df['LS_ACTUAL_DATE'] <= pd.Timestamp(end_date))]

    # Map coordinates for LEGO_ZONE_DESC and LEGD_ZONE_DESC
    df = df.merge(city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"}),
                  on="LEGO_ZONE_DESC", how="left")
    df = df.merge(city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"}),
                  on="LEGD_ZONE_DESC", how="left")
    return df

@st.cache_data
def preprocess_tlorder(file):
    df = pd.read_csv(file, low_memory=False)
    df.rename(columns={"BILL_NUMBER": "TLORDER_BILL_NUMBER"}, inplace=True)
    return df

@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df.rename(columns={"BILL_NUMBER": "DRIVERPAY_BILL_NUMBER"}, inplace=True)
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


if uploaded_legsum_file and uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)

    # Merge DRIVERPAY data
    driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)
    legsum_df = legsum_df.merge(driver_pay_agg, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Add currency conversion for charges (if applicable)
    exchange_rate = 1.38

    # Ensure CHARGES and XCHARGES are numeric, replacing invalid entries with NaN
    legsum_df['CHARGES'] = pd.to_numeric(legsum_df.get('CHARGES', None), errors='coerce')
    legsum_df['XCHARGES'] = pd.to_numeric(legsum_df.get('XCHARGES', None), errors='coerce')

    # Calculate TOTAL_CHARGE_CAD only for rows where BILL_NUMBER exists
    legsum_df['TOTAL_CHARGE_CAD'] = np.where(
        pd.notna(legsum_df['BILL_NUMBER']),  # Check if BILL_NUMBER exists
        (legsum_df['CHARGES'].fillna(0) + legsum_df['XCHARGES'].fillna(0)) * exchange_rate,  # Sum charges and convert
        None  # Set to None if BILL_NUMBER is missing
    )

    # Ensure LS_LEG_DIST is numeric and handle zeros explicitly
    legsum_df['LS_LEG_DIST'] = pd.to_numeric(legsum_df['LS_LEG_DIST'], errors='coerce')
    legsum_df['LS_LEG_DIST'] = np.where(legsum_df['LS_LEG_DIST'] > 0, legsum_df['LS_LEG_DIST'], np.nan)

    # Calculate Revenue per Mile safely, only if LS_LEG_DIST > 0
    legsum_df['Revenue per Mile'] = np.where(
        pd.notna(legsum_df['TOTAL_CHARGE_CAD']) & pd.notna(legsum_df['LS_LEG_DIST']),
        legsum_df['TOTAL_CHARGE_CAD'] / legsum_df['LS_LEG_DIST'],  # Calculate RPM
        np.nan  # Assign NaN if distance is zero or TOTAL_CHARGE_CAD is missing
    )

    # Calculate Profit (CAD) only if TOTAL_CHARGE_CAD is available
    legsum_df['Profit (CAD)'] = np.where(
        pd.notna(legsum_df['TOTAL_CHARGE_CAD']),
        legsum_df['TOTAL_CHARGE_CAD'] - legsum_df['TOTAL_PAY_AMT'].fillna(0),  # Calculate Profit
        np.nan  # Assign NaN if TOTAL_CHARGE_CAD is missing
    )

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

    # Ensure LS_POWER_UNIT is treated as a string
    legsum_df['LS_POWER_UNIT'] = legsum_df['LS_POWER_UNIT'].astype(str)
    
    # Create options for selecting Power Unit
    punit_options = sorted(legsum_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
    
    # Filter relevant drivers for the selected Power Unit
    relevant_drivers = legsum_df[legsum_df['LS_POWER_UNIT'] == selected_punit]['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    
    # Filter data based on the selected Power Unit and optionally Driver ID
    filtered_view = legsum_df[legsum_df['LS_POWER_UNIT'] == selected_punit].copy()
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()
    
    # Filter data by month if data exists
    months = sorted(filtered_view['Month'].unique())
    if len(months) == 0:
        st.warning("No data available for the selected Power Unit and Driver ID.")
    else:
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()
    
    # Process and summarize data if available for the selected month
    if not month_data.empty:
        # Assign colors for alternating rows by day
        month_data = month_data.sort_values(by='LS_ACTUAL_DATE')
        month_data['Day_Group'] = month_data['LS_ACTUAL_DATE'].dt.date
        unique_days = list(month_data['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        month_data['Highlight'] = month_data['Day_Group'].map(day_colors)
    
        # Add profit calculation
        month_data['Profit (CAD)'] = month_data['TOTAL_CHARGE_CAD'] - month_data['TOTAL_PAY_AMT']
    
        # Create the route summary DataFrame
        route_summary_df = month_data.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[
            ["Route", "LS_FREIGHT", "TOTAL_CHARGE_CAD", "LS_LEG_DIST", "Straight Distance",
             "Revenue per Mile", "DRIVER_ID", "TOTAL_PAY_AMT", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight"]
        ].rename(columns={
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
    st.warning("Please upload LEGSUM, TLORDER, and DRIVERPAY CSV files to proceed.")
  


