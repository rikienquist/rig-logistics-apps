import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "date_range" not in st.session_state:
    st.session_state.date_range = {"start_date": None, "end_date": None}

# Streamlit App Title and Instructions
st.title("Trip Map Viewer")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_DRIVER, LS_FREIGHT, LS_TRIP_NUMBER, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y;

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
def preprocess_legsum(file):
    """Preprocess the LEGSUM file to standardize and clean the data."""
    df = pd.read_csv(file, low_memory=False, parse_dates=["LS_ACTUAL_DATE"])
    return df


@st.cache_data
def preprocess_tlorder_driverpay(file):
    """Preprocess the TLORDER + DRIVERPAY file to standardize and clean the data."""
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_SUM'] = pd.to_numeric(df['TOTAL_PAY_SUM'], errors='coerce')
    return df


@st.cache_data
def enrich_coordinates(df, city_coords):
    """Enrich the LEGSUM data with coordinates for LEGO_ZONE_DESC and LEGD_ZONE_DESC."""
    # Clean and standardize location names
    city_coords['LOCATION'] = city_coords['LOCATION'].str.strip().str.upper()

    # Merge coordinates for LEGO_ZONE_DESC
    origin_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(origin_coords, on="LEGO_ZONE_DESC", how="left")

    # Merge coordinates for LEGD_ZONE_DESC
    dest_coords = city_coords.rename(columns={"LEGO_ZONE_DESC": "LEGD_ZONE_DESC", "ORIG_LAT": "DEST_LAT", "ORIG_LON": "DEST_LON"})
    df = df.merge(dest_coords, on="LEGD_ZONE_DESC", how="left")

    return df


@st.cache_data
def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculate the straight-line distance between two geographic coordinates."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


@st.cache_data
def calculate_distances(df):
    """Calculate distances (straight-line) for all routes in the dataset."""
    df['Straight Distance (miles)'] = calculate_haversine(
        df['ORIG_LAT'], df['ORIG_LON'], df['DEST_LAT'], df['DEST_LON']
    )
    return df
if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Merge LEGSUM with TLORDER+DRIVERPAY on LS_FREIGHT and BILL_NUMBER
    merged_df = legsum_df.merge(tlorder_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Enrich with coordinates for LEGO_ZONE_DESC and LEGD_ZONE_DESC
    enriched_df = enrich_coordinates(merged_df, city_coordinates_df)

    # Filter for rows with valid routes (both origin and destination must exist)
    valid_routes = enriched_df[
        (pd.notna(enriched_df['LEGO_ZONE_DESC']) & pd.notna(enriched_df['LEGD_ZONE_DESC']))
    ].copy()

    # Convert charges to CAD (if currency is USD)
    exchange_rate = 1.38
    valid_routes['CHARGES'] = pd.to_numeric(valid_routes['CHARGES'], errors='coerce')
    valid_routes['XCHARGES'] = pd.to_numeric(valid_routes['XCHARGES'], errors='coerce')
    valid_routes['TOTAL_CHARGE_CAD'] = np.where(
        valid_routes['CURRENCY_CODE'] == 'USD',
        (valid_routes['CHARGES'] + valid_routes['XCHARGES']) * exchange_rate,
        valid_routes['CHARGES'] + valid_routes['XCHARGES']
    )

    # Filter out rows with zero or missing charges
    filtered_df = valid_routes[
        (valid_routes['TOTAL_CHARGE_CAD'] > 0)
    ].copy()

    # Calculate revenue per mile and profit
    filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['LS_LEG_DIST']
    filtered_df['Profit (CAD)'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_SUM']

    # Add straight-line distance using haversine formula
    filtered_df = calculate_distances(filtered_df)

    # Find missing locations in coordinates
    missing_origins = filtered_df[
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['ORIG_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})

    missing_destinations = filtered_df[
        pd.isna(filtered_df['DEST_LAT']) | pd.isna(filtered_df['DEST_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Create options for LS_POWER_UNIT and LS_DRIVER
    unit_options = sorted(filtered_df['LS_POWER_UNIT'].unique())
    selected_unit = st.selectbox("Select Power Unit:", options=unit_options)
    
    relevant_drivers = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_unit]['LS_DRIVER'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    
    # Filter data based on selected LS_POWER_UNIT and optionally LS_DRIVER
    filtered_view = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_unit].copy()
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['LS_DRIVER'] == selected_driver].copy()
    
    # Allow the user to filter data by date range (LS_ACTUAL_DATE)
    st.sidebar.header("Date Range Selection")
    start_date = st.sidebar.date_input("Start Date", value=filtered_view['LS_ACTUAL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=filtered_view['LS_ACTUAL_DATE'].max())
    filtered_view = filtered_view[
        (filtered_view['LS_ACTUAL_DATE'] >= pd.to_datetime(start_date)) &
        (filtered_view['LS_ACTUAL_DATE'] <= pd.to_datetime(end_date))
    ]
    
    if filtered_view.empty:
        st.warning("No data available for the selected Power Unit and Driver ID within the specified date range.")
    else:
        # Assign colors for alternating rows by day
        filtered_view = filtered_view.sort_values(by='LS_ACTUAL_DATE')
        filtered_view['Day_Group'] = filtered_view['LS_ACTUAL_DATE'].dt.date
        unique_days = list(filtered_view['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        filtered_view['Highlight'] = filtered_view['Day_Group'].map(day_colors)
    
        # Create the route summary DataFrame
        filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_SUM']
    
        route_summary_df = filtered_view.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[[  # Include "Highlight" for styling
            "Route", "BILL_NUMBER", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED",
            "Total Charge (CAD)", "Straight Distance (miles)", "Revenue per Mile",
            "LS_DRIVER", "TOTAL_PAY_SUM", "Profit (CAD)", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Highlight"
        ]].rename(columns={
            "TOTAL_PAY_SUM": "Driver Pay (CAD)",
            "LS_ACTUAL_DATE": "Date"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "LS_TRIP_NUMBER": "",
            "LS_LEG_DIST": route_summary_df["LS_LEG_DIST"].sum(),
            "LS_MT_LOADED": route_summary_df["LS_MT_LOADED"].sum(),
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Straight Distance (miles)": route_summary_df["Straight Distance (miles)"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Straight Distance (miles)"].sum()
            if route_summary_df["Straight Distance (miles)"].sum() != 0 else 0,
            "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "Date": "",
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

        # Generate the map
        fig = go.Figure()
        
        # Assign unique sequences for LEGO_ZONE_DESC and LEGD_ZONE_DESC
        location_sequence = {location: [] for location in set(filtered_view['LEGO_ZONE_DESC']).union(filtered_view['LEGD_ZONE_DESC'])}
        label_counter = 1
        for _, row in filtered_view.iterrows():
            location_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
            label_counter += 1
            location_sequence[row['LEGD_ZONE_DESC']].append(label_counter)
            label_counter += 1
        
        # Legend flags
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        for _, row in filtered_view.iterrows():
            origin_sequence = ", ".join(map(str, location_sequence[row['LEGO_ZONE_DESC']]))
            destination_sequence = ", ".join(map(str, location_sequence[row['LEGD_ZONE_DESC']]))
        
            # Get aggregated values for origin location
            total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGO_ZONE_DESC'], row['ORIG_LAT'])
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
            total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGD_ZONE_DESC'], row['DEST_LAT'])
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
        
        # Update map layout
        fig.update_layout(
            title=f"Routes for Power Unit: {selected_unit}, Driver ID: {selected_driver}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        
        st.plotly_chart(fig)
        
        # Display missing locations (relevant to the current filtered view)
        missing_origins = filtered_view[
            pd.isna(filtered_view['ORIG_LAT']) | pd.isna(filtered_view['ORIG_LON'])
        ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Location'})
        
        missing_destinations = filtered_view[
            pd.isna(filtered_view['DEST_LAT']) | pd.isna(filtered_view['DEST_LON'])
        ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Location'})
        
        missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()
        
        if not missing_locations.empty:
            st.write("### Locations Missing Coordinates")
            st.dataframe(missing_locations, use_container_width=True)

    else:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    
else:
    st.warning("Please upload both CSV files to proceed.")




