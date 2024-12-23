### driver pay works but not TLORDER

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

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

uploaded_legsum_file = st.file_uploader("Upload LEGSUM CSV file", type="csv")
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file (optional)", type="csv")

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
def preprocess_legsum(file, coordinates):
    legsum_df = pd.read_csv(file)
    legsum_df = legsum_df.rename(columns={
        "LEGO_ZONE_DESC": "ORIG_LOCATION",
        "LEGD_ZONE_DESC": "DEST_LOCATION"
    })
    
    # Merge coordinates for origins and destinations
    origin_coords = coordinates.rename(columns={"LOCATION": "ORIG_LOCATION", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    legsum_df = legsum_df.merge(origin_coords, on="ORIG_LOCATION", how="left")

    dest_coords = coordinates.rename(columns={"LOCATION": "DEST_LOCATION", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    legsum_df = legsum_df.merge(dest_coords, on="DEST_LOCATION", how="left")

    return legsum_df

@st.cache_data
def preprocess_tlorder(file):
    return pd.read_csv(file)

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_tlorder_file and uploaded_legsum_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file)

    # Merge LEGSUM and TLORDER data on LS_FREIGHT and BILL_NUMBER
    merged_df = legsum_df.merge(
        tlorder_df[['BILL_NUMBER', 'CHARGES', 'XCHARGES', 'CURRENCY_CODE']],
        left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left'
    )

    # Ensure numeric fields are parsed correctly
    merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
    merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')

    # Calculate Total Charge in CAD
    exchange_rate = 1.38
    merged_df['Total Charge (CAD)'] = np.where(
        merged_df['CURRENCY_CODE'] == 'USD',
        (merged_df['CHARGES'] + merged_df['XCHARGES']) * exchange_rate,
        merged_df['CHARGES'] + merged_df['XCHARGES']
    )

    # Filter for valid routes: non-zero charges, and valid distances
    valid_routes = merged_df[
        (merged_df['Total Charge (CAD)'] > 0) &
        (merged_df['LS_LEG_DIST'] > 0)  # Ensure positive leg distances
    ].copy()

    # Calculate additional metrics: Revenue per Mile, Profit
    valid_routes['Revenue per Mile'] = valid_routes['Total Charge (CAD)'] / valid_routes['LS_LEG_DIST']
    valid_routes['Profit (CAD)'] = valid_routes['Total Charge (CAD)'] - valid_routes['LS_MT_LOADED']

    # Calculate Straight Distance using coordinates
    valid_routes['Straight Distance'] = np.where(
        pd.isna(valid_routes['ORIG_LAT']) | pd.isna(valid_routes['DEST_LAT']),
        np.nan,
        calculate_haversine(valid_routes)
    )

    # Handle missing locations (origins or destinations with missing coordinates)
    missing_origins = valid_routes[
        pd.isna(valid_routes['ORIG_LAT']) | pd.isna(valid_routes['ORIG_LON'])
    ][['ORIG_LOCATION']].drop_duplicates().rename(columns={'ORIG_LOCATION': 'Location'})

    missing_destinations = valid_routes[
        pd.isna(valid_routes['DEST_LAT']) | pd.isna(valid_routes['DEST_LON'])
    ][['DEST_LOCATION']].drop_duplicates().rename(columns={'DEST_LOCATION': 'Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Display warnings or missing locations if any
    if not missing_locations.empty:
        st.write("### Missing Locations")
        st.dataframe(missing_locations, use_container_width=True)

    # Processed and filtered dataframe is ready for further analysis or visualization
    st.write("### Valid Routes")
    st.dataframe(valid_routes, use_container_width=True)

    # Filter data based on LS_POWER_UNIT
    power_unit_options = sorted(filtered_df['LS_POWER_UNIT'].unique())
    selected_power_unit = st.selectbox("Select Power Unit:", options=power_unit_options)
    
    # Filter based on selected Power Unit
    filtered_view = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_power_unit].copy()
    
    # Select start and end dates for LS_ACTUAL_DATE
    start_date = st.date_input("Start Date:", value=filtered_view['LS_ACTUAL_DATE'].min())
    end_date = st.date_input("End Date:", value=filtered_view['LS_ACTUAL_DATE'].max())
    
    # Further filter data based on selected dates
    filtered_view = filtered_view[
        (pd.to_datetime(filtered_view['LS_ACTUAL_DATE']) >= pd.Timestamp(start_date)) &
        (pd.to_datetime(filtered_view['LS_ACTUAL_DATE']) <= pd.Timestamp(end_date))
    ]
    
    if filtered_view.empty:
        st.warning("No data available for the selected Power Unit and date range.")
    else:
        # Create the route summary DataFrame
        filtered_view['Profit (CAD)'] = filtered_view['Total Charge (CAD)'] - filtered_view['LS_MT_LOADED']
        filtered_view['Revenue per Mile'] = filtered_view['Total Charge (CAD)'] / filtered_view['LS_LEG_DIST']
        
        route_summary_df = filtered_view.assign(
            Route=lambda x: x['ORIG_LOCATION'] + " to " + x['DEST_LOCATION']
        )[
            ["Route", "LS_FREIGHT", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED", 
             "Total Charge (CAD)", "Straight Distance", "Revenue per Mile", "Profit (CAD)", 
             "LS_ACTUAL_DATE", "LS_LEG_NOTE"]
        ].rename(columns={
            "LS_FREIGHT": "BILL_NUMBER",
            "LS_TRIP_NUMBER": "Trip Number",
            "LS_LEG_DIST": "Distance (miles)",
            "LS_MT_LOADED": "Driver Pay (CAD)",
            "Straight Distance": "Straight Distance (miles)"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Trip Number": "",
            "Distance (miles)": route_summary_df["Distance (miles)"].sum(),
            "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Straight Distance (miles)": route_summary_df["Straight Distance (miles)"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Distance (miles)"].sum()
            if route_summary_df["Distance (miles)"].sum() != 0 else 0,
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "LS_ACTUAL_DATE": "",
            "LS_LEG_NOTE": ""
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
            else:
                return ['background-color: #f7f7f7'] * len(row)
    
        styled_route_summary = route_summary_df.style.apply(highlight_rows, axis=1)
        st.write("Route Summary:")
        st.dataframe(styled_route_summary, use_container_width=True)
    
        # Combine origins and destinations for aggregated totals
        city_aggregates = pd.concat([
            filtered_view[['ORIG_LOCATION', 'Total Charge (CAD)', 'LS_LEG_DIST', 'LS_MT_LOADED']].rename(
                columns={'ORIG_LOCATION': 'City'}
            ),
            filtered_view[['DEST_LOCATION', 'Total Charge (CAD)', 'LS_LEG_DIST', 'LS_MT_LOADED']].rename(
                columns={'DEST_LOCATION': 'City'}
            )
        ], ignore_index=True)
    
        city_aggregates = city_aggregates.groupby('City', as_index=False).agg({
            'Total Charge (CAD)': 'sum',
            'LS_LEG_DIST': 'sum',
            'LS_MT_LOADED': 'sum'
        })
    
        city_aggregates['Revenue per Mile'] = city_aggregates['Total Charge (CAD)'] / city_aggregates['LS_LEG_DIST']
        city_aggregates['Profit (CAD)'] = city_aggregates['Total Charge (CAD)'] - city_aggregates['LS_MT_LOADED']
    
        st.write("City Aggregates:")
        st.dataframe(city_aggregates, use_container_width=True)

        # Generate the map
        fig = go.Figure()
        
        # Track the sequence of cities
        city_sequence = {location: [] for location in set(filtered_view['ORIG_LOCATION']).union(filtered_view['DEST_LOCATION'])}
        label_counter = 1
        for _, row in filtered_view.iterrows():
            city_sequence[row['ORIG_LOCATION']].append(label_counter)
            label_counter += 1
            city_sequence[row['DEST_LOCATION']].append(label_counter)
            label_counter += 1
        
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        for _, row in filtered_view.iterrows():
            origin_sequence = ", ".join(map(str, city_sequence[row['ORIG_LOCATION']]))
            destination_sequence = ", ".join(map(str, city_sequence[row['DEST_LOCATION']]))
        
            # Get aggregated values for origin location
            total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['ORIG_LOCATION'], row['DEST_LOCATION'])
            hover_origin_text = (
                f"Location: {row['ORIG_LOCATION']}<br>"
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
            hover_dest_text = (
                f"Location: {row['DEST_LOCATION']}<br>"
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
            title=f"Routes for {start_date} to {end_date} - Power Unit: {selected_power_unit}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)
        
        # Display locations missing coordinates relevant to the selection
        missing_origins = filtered_view[
            (pd.isna(filtered_view['ORIG_LAT']) | pd.isna(filtered_view['ORIG_LON']))
        ][['ORIG_LOCATION']].drop_duplicates().rename(columns={'ORIG_LOCATION': 'Location'})
        
        missing_destinations = filtered_view[
            (pd.isna(filtered_view['DEST_LAT']) | pd.isna(filtered_view['DEST_LON']))
        ][['DEST_LOCATION']].drop_duplicates().rename(columns={'DEST_LOCATION': 'Location'})
        
        missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()
        
        if not missing_locations.empty:
            st.write("### Missing Locations")
            st.dataframe(missing_locations, use_container_width=True)












