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
def preprocess_files(legsumm_file, tlorder_file, city_coords):
    legsumm_df = pd.read_csv(legsumm_file, low_memory=False)
    tlorder_df = pd.read_csv(tlorder_file, low_memory=False)
    
    # Merge LEGSUM with TLORDER using LS_FREIGHT to BILL_NUMBER
    merged_df = legsumm_df.merge(tlorder_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Clean and standardize location names
    def clean_location_name(name):
        return re.sub(r"[^a-zA-Z0-9\s]", "", str(name)).strip().upper()

    city_coords['LOCATION'] = city_coords['LOCATION'].apply(clean_location_name)
    merged_df['LEGO_ZONE_DESC'] = merged_df['LEGO_ZONE_DESC'].apply(clean_location_name)
    merged_df['LEGD_ZONE_DESC'] = merged_df['LEGD_ZONE_DESC'].apply(clean_location_name)

    # Merge for LEGO_ZONE_DESC
    lego_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "LEGO_LAT", "LON": "LEGO_LON"})
    merged_df = merged_df.merge(lego_coords, on="LEGO_ZONE_DESC", how="left")

    # Merge for LEGD_ZONE_DESC
    legd_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "LEGD_LAT", "LON": "LEGD_LON"})
    merged_df = merged_df.merge(legd_coords, on="LEGD_ZONE_DESC", how="left")

    return merged_df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = np.radians(df['LEGO_LAT']), np.radians(df['LEGO_LON'])
    lat2, lon2 = np.radians(df['LEGD_LAT']), np.radians(df['LEGD_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file, city_coordinates_df)
    driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)

    # Merge TLORDER and DRIVERPAY data
    tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

    # Ensure numeric columns are correctly parsed
    tlorder_df['DISTANCE'] = pd.to_numeric(tlorder_df['DISTANCE'], errors='coerce')
    tlorder_df['CHARGES'] = pd.to_numeric(tlorder_df['CHARGES'], errors='coerce')
    tlorder_df['XCHARGES'] = pd.to_numeric(tlorder_df['XCHARGES'], errors='coerce')

    # Filter for valid routes (non-identical origins and destinations, with a valid distance)
    valid_routes = tlorder_df[
        (tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) &
        (pd.notna(tlorder_df['DISTANCE']))
    ].copy()

    # Calculate total charges in CAD
    exchange_rate = 1.38
    valid_routes['TOTAL_CHARGE_CAD'] = np.where(
        valid_routes['CURRENCY_CODE'] == 'USD',
        (valid_routes['CHARGES'] + valid_routes['XCHARGES']) * exchange_rate,
        valid_routes['CHARGES'] + valid_routes['XCHARGES']
    )

    # Filter for non-zero total charges
    filtered_df = valid_routes[valid_routes['TOTAL_CHARGE_CAD'] > 0].copy()

    # Add calculated fields
    filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
    filtered_df['Profit (CAD)'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_AMT']

    # Identify missing coordinates for origins and destinations
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

    # Combine missing cities into a single dataframe
    missing_cities = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Calculate Straight Distance using the Haversine formula, where coordinates exist
    filtered_df['Straight Distance'] = np.where(
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['DEST_LAT']),
        np.nan,
        calculate_haversine(filtered_df)
    )

    # Filter based on selected truck unit (LS_POWER_UNIT)
    punit_options = sorted(filtered_df['LS_POWER_UNIT'].unique())
    selected_punit = st.selectbox("Select Truck Unit (LS_POWER_UNIT):", options=punit_options)
    
    # Filter data based on selected truck unit
    filtered_view = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit].copy()
    
    # Date range selection
    start_date = st.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp("2024-12-31"))
    filtered_view = filtered_view[
        (filtered_view['LS_ACTUAL_DATE'] >= pd.Timestamp(start_date)) &
        (filtered_view['LS_ACTUAL_DATE'] <= pd.Timestamp(end_date))
    ]
    
    if filtered_view.empty:
        st.warning("No data available for the selected Truck Unit and date range.")
    else:
        # Assign colors for alternating rows by day
        filtered_view = filtered_view.sort_values(by='LS_ACTUAL_DATE')
        filtered_view['Day_Group'] = filtered_view['LS_ACTUAL_DATE'].dt.date
        unique_days = list(filtered_view['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        filtered_view['Highlight'] = filtered_view['Day_Group'].map(day_colors)
        
        # Create the route summary DataFrame
        filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_AMT']
        route_summary_df = filtered_view.assign(
            Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
        )[[  # Include "Highlight" for styling
            "Route", "BILL_NUMBER", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED", 
            "Total Charge (CAD)", "LS_LEG_DIST", "Straight Distance", 
            "Revenue per Mile", "LS_ACTUAL_DATE", "LS_LEG_NOTE", "Profit (CAD)", "Highlight"
        ]].rename(columns={
            "LS_TRIP_NUMBER": "Trip Number",
            "LS_LEG_DIST": "Distance (miles)",
            "LS_MT_LOADED": "MT Loaded",
            "LS_ACTUAL_DATE": "Actual Date",
            "LS_LEG_NOTE": "Leg Note"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Trip Number": "",
            "Distance (miles)": route_summary_df["Distance (miles)"].sum(),
            "MT Loaded": route_summary_df["MT Loaded"].sum(),
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Straight Distance": route_summary_df["Straight Distance"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Distance (miles)"].sum()
            if route_summary_df["Distance (miles)"].sum() != 0 else 0,
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "Actual Date": "",
            "Leg Note": "",
            "Highlight": None
        }])
    
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
        
        # Format currency columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Profit (CAD)"]:
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
        city_aggregates = pd.concat([
            filtered_view[['LEGO_ZONE_DESC', 'LEGO_LAT', 'LEGO_LON', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGO_ZONE_DESC': 'City'}
            ),
            filtered_view[['LEGD_ZONE_DESC', 'LEGD_LAT', 'LEGD_LON', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_AMT']].rename(
                columns={'LEGD_ZONE_DESC': 'City'}
            )
        ], ignore_index=True)
        
        # Clean and aggregate the combined data
        city_aggregates = city_aggregates.groupby(['City'], as_index=False).agg({
            'TOTAL_CHARGE_CAD': 'sum',
            'LS_LEG_DIST': 'sum',
            'TOTAL_PAY_AMT': 'sum'
        })
        city_aggregates['Revenue per Mile'] = city_aggregates['TOTAL_CHARGE_CAD'] / city_aggregates['LS_LEG_DIST']
        city_aggregates['Profit (CAD)'] = city_aggregates['TOTAL_CHARGE_CAD'] - city_aggregates['TOTAL_PAY_AMT']
    
        # Generate the map
        fig = go.Figure()
        
        # Create a sequence for labeling cities
        city_sequence = {city: [] for city in set(filtered_view['LEGO_ZONE_DESC']).union(filtered_view['LEGD_ZONE_DESC'])}
        label_counter = 1
        for _, row in filtered_view.iterrows():
            city_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
            label_counter += 1
            city_sequence[row['LEGD_ZONE_DESC']].append(label_counter)
            label_counter += 1
        
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        # Iterate through the filtered data to create map elements
        for _, row in filtered_view.iterrows():
            origin_sequence = ", ".join(map(str, city_sequence[row['LEGO_ZONE_DESC']]))
            destination_sequence = ", ".join(map(str, city_sequence[row['LEGD_ZONE_DESC']]))
        
            # Get aggregated values for the origin city
            total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGO_ZONE_DESC'], row['LEGO_LAT'])
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
        
            # Get aggregated values for the destination city
            total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGD_ZONE_DESC'], row['LEGD_LAT'])
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
            title=f"Routes for LS_POWER_UNIT: {selected_punit}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)
        
        # Display missing locations relevant to the selection
        relevant_missing_origins = filtered_view[
            pd.isna(filtered_view['LEGO_LAT']) | pd.isna(filtered_view['LEGO_LON'])
        ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={
            'LEGO_ZONE_DESC': 'Location'
        })
        
        relevant_missing_destinations = filtered_view[
            pd.isna(filtered_view['LEGD_LAT']) | pd.isna(filtered_view['LEGD_LON'])
        ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={
            'LEGD_ZONE_DESC': 'Location'
        })
        
        relevant_missing_locations = pd.concat([relevant_missing_origins, relevant_missing_destinations]).drop_duplicates()
        
        if not relevant_missing_locations.empty:
            st.write("### Missing Locations")
            st.dataframe(relevant_missing_locations, use_container_width=True)


else:
    st.warning("Please upload both the LEGSUM and TLORDER CSV files to proceed.")







