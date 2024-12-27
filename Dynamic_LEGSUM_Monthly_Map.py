import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "date_range" not in st.session_state:
    st.session_state.date_range = None

# Streamlit App Title and Instructions
st.title("Trip Map Viewer by Date Range")

st.markdown("""
### Instructions:
Use the following query to generate the required LEGSUM data:  
SELECT LS_POWER_UNIT, LS_DRIVER, LS_FREIGHT, LS_TRIP_NUMBER, LEGO_ZONE_DESC, LEGD_ZONE_DESC, 
       LS_LEG_DIST, LS_MT_LOADED, LS_ACTUAL_DATE, LS_LEG_NOTE  
FROM LEGSUM WHERE "LS_ACTUAL_DATE" BETWEEN 'X' AND 'Y';

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
uploaded_tl_driverpay_file = st.file_uploader("Upload TLORDER + DRIVERPAY CSV file", type="csv")

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
    legsum_df = pd.read_csv(file, low_memory=False)
    legsum_df['LS_ACTUAL_DATE'] = pd.to_datetime(legsum_df['LS_ACTUAL_DATE'])
    
    city_coords['LOCATION'] = city_coords['LOCATION'].str.strip().str.upper()
    legsum_df['LEGO_ZONE_DESC'] = legsum_df['LEGO_ZONE_DESC'].str.strip().str.upper()
    legsum_df['LEGD_ZONE_DESC'] = legsum_df['LEGD_ZONE_DESC'].str.strip().str.upper()
    
    # Merge for LEGO_ZONE_DESC
    origin_coords = city_coords.rename(columns={"LOCATION": "LEGO_ZONE_DESC", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    legsum_df = legsum_df.merge(origin_coords, on="LEGO_ZONE_DESC", how="left")
    
    # Merge for LEGD_ZONE_DESC
    dest_coords = city_coords.rename(columns={"LOCATION": "LEGD_ZONE_DESC", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    legsum_df = legsum_df.merge(dest_coords, on="LEGD_ZONE_DESC", how="left")
    
    return legsum_df

@st.cache_data
def preprocess_tl_driverpay(file):
    tl_driverpay_df = pd.read_csv(file, low_memory=False)
    tl_driverpay_df['TOTAL_PAY_SUM'] = pd.to_numeric(tl_driverpay_df['TOTAL_PAY_SUM'], errors='coerce')
    return tl_driverpay_df

@st.cache_data
def calculate_haversine(df):
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_legsum_file and uploaded_tl_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file, city_coordinates_df)
    tl_driverpay_df = preprocess_tl_driverpay(uploaded_tl_driverpay_file)

    # Merge LEGSUM and TLORDER+DRIVERPAY data
    merged_df = legsum_df.merge(tl_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # Calculate Total Charges in CAD
    exchange_rate = 1.38
    merged_df['CHARGES'] = pd.to_numeric(merged_df['CHARGES'], errors='coerce')
    merged_df['XCHARGES'] = pd.to_numeric(merged_df['XCHARGES'], errors='coerce')
    merged_df['TOTAL_CHARGE_CAD'] = np.where(
        merged_df['CURRENCY_CODE'] == 'USD',
        (merged_df['CHARGES'] + merged_df['XCHARGES']) * exchange_rate,
        merged_df['CHARGES'] + merged_df['XCHARGES']
    )

    # Filter for valid routes (exclude rows with same origin and destination or missing distance)
    valid_routes = merged_df[
        (merged_df['LEGO_ZONE_DESC'] != merged_df['LEGD_ZONE_DESC']) &
        pd.notna(merged_df['LS_LEG_DIST'])
    ].copy()

    # Compute Revenue per Mile and Profit
    valid_routes['Revenue per Mile'] = valid_routes['TOTAL_CHARGE_CAD'] / valid_routes['LS_LEG_DIST']
    valid_routes['Profit (CAD)'] = valid_routes['TOTAL_CHARGE_CAD'] - valid_routes['TOTAL_PAY_SUM']

    # Filter by user-selected date range
    start_date = st.date_input("Start Date", value=pd.Timestamp.today() - pd.Timedelta(days=30))
    end_date = st.date_input("End Date", value=pd.Timestamp.today())
    filtered_df = valid_routes[
        (valid_routes['LS_ACTUAL_DATE'] >= start_date) & (valid_routes['LS_ACTUAL_DATE'] <= end_date)
    ]

    # Find locations missing in the coordinates dataset
    missing_origins = filtered_df[
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['ORIG_LON'])
    ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={'LEGO_ZONE_DESC': 'Missing Location'})

    missing_destinations = filtered_df[
        pd.isna(filtered_df['DEST_LAT']) | pd.isna(filtered_df['DEST_LON'])
    ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={'LEGD_ZONE_DESC': 'Missing Location'})

    missing_locations = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Calculate Straight Distance
    filtered_df['Straight Distance (miles)'] = np.where(
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['DEST_LAT']),
        np.nan,
        calculate_haversine(filtered_df)
    )

       punit_options = sorted(filtered_df['LS_POWER_UNIT'].unique())
       selected_punit = st.selectbox("Select Power Unit:", options=punit_options)
       
       relevant_drivers = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit]['LS_DRIVER'].unique()
       driver_options = ["All"] + sorted(relevant_drivers.astype(str))
       selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
       
       filtered_view = filtered_df[filtered_df['LS_POWER_UNIT'] == selected_punit].copy()
       if selected_driver != "All":
           filtered_view = filtered_view[filtered_view['LS_DRIVER'] == selected_driver].copy()
       
       if filtered_view.empty:
           st.warning("No data available for the selected Power Unit and Driver ID.")
       else:
           # Sort by actual date
           filtered_view = filtered_view.sort_values(by='LS_ACTUAL_DATE')
           filtered_view['Day_Group'] = filtered_view['LS_ACTUAL_DATE'].dt.date
           unique_days = list(filtered_view['Day_Group'].unique())
           day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
           filtered_view['Highlight'] = filtered_view['Day_Group'].map(day_colors)
       
           # Calculate profit
           filtered_view['Profit (CAD)'] = filtered_view['TOTAL_CHARGE_CAD'] - filtered_view['TOTAL_PAY_SUM']
       
           # Create route summary DataFrame
           route_summary_df = filtered_view.assign(
               Route=lambda x: x['LEGO_ZONE_DESC'] + " to " + x['LEGD_ZONE_DESC']
           )[
               [
                   "Route", "BILL_NUMBER", "LS_TRIP_NUMBER", "LS_LEG_DIST", "LS_MT_LOADED",
                   "TOTAL_CHARGE_CAD", "Straight Distance (miles)", "Revenue per Mile",
                   "LS_DRIVER", "TOTAL_PAY_SUM", "Profit (CAD)", "LS_ACTUAL_DATE", "Highlight"
               ]
           ].rename(columns={
               "TOTAL_CHARGE_CAD": "Total Charge (CAD)",
               "LS_LEG_DIST": "Distance (miles)",
               "Straight Distance (miles)": "Straight Distance (miles)",
               "TOTAL_PAY_SUM": "Driver Pay (CAD)"
           })
       
           # Calculate grand totals
           grand_totals = pd.DataFrame([{
               "Route": "Grand Totals",
               "BILL_NUMBER": "",
               "LS_TRIP_NUMBER": "",
               "Distance (miles)": route_summary_df["Distance (miles)"].sum(),
               "LS_MT_LOADED": "",
               "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
               "Straight Distance (miles)": route_summary_df["Straight Distance (miles)"].sum(),
               "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Distance (miles)"].sum()
               if route_summary_df["Distance (miles)"].sum() != 0 else 0,
               "LS_DRIVER": "",
               "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
               "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
               "LS_ACTUAL_DATE": "",
               "Highlight": None
           }])
       
           route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
       
           # Format currency and numeric columns
           for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
               route_summary_df[col] = route_summary_df[col].apply(
                   lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
               )
       
           # Highlight rows for display
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
               filtered_view[['LEGO_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_SUM']].rename(
                   columns={'LEGO_ZONE_DESC': 'Location'}
               ),
               filtered_view[['LEGD_ZONE_DESC', 'TOTAL_CHARGE_CAD', 'LS_LEG_DIST', 'TOTAL_PAY_SUM']].rename(
                   columns={'LEGD_ZONE_DESC': 'Location'}
               )
           ], ignore_index=True)
       
           # Aggregate by location
           location_aggregates = location_aggregates.groupby('Location', as_index=False).agg({
               'TOTAL_CHARGE_CAD': 'sum',
               'LS_LEG_DIST': 'sum',
               'TOTAL_PAY_SUM': 'sum'
           })
           location_aggregates['Revenue per Mile'] = location_aggregates['TOTAL_CHARGE_CAD'] / location_aggregates['LS_LEG_DIST']
           location_aggregates['Profit (CAD)'] = location_aggregates['TOTAL_CHARGE_CAD'] - location_aggregates['TOTAL_PAY_SUM']
       
           # Display aggregated data
           st.write("Location Aggregates:")
           st.dataframe(location_aggregates, use_container_width=True)
       
       # Generate the map
       fig = go.Figure()
       
       location_sequence = {location: [] for location in set(filtered_view['LEGO_ZONE_DESC']).union(filtered_view['LEGD_ZONE_DESC'])}
       label_counter = 1
       for _, row in filtered_view.iterrows():
           location_sequence[row['LEGO_ZONE_DESC']].append(label_counter)
           label_counter += 1
           location_sequence[row['LEGD_ZONE_DESC']].append(label_counter)
           label_counter += 1
       
       legend_added = {"Origin": False, "Destination": False, "Route": False}
       
       for _, row in filtered_view.iterrows():
           origin_sequence = ", ".join(map(str, location_sequence[row['LEGO_ZONE_DESC']]))
           destination_sequence = ", ".join(map(str, location_sequence[row['LEGD_ZONE_DESC']]))
       
           # Get aggregated values for origin location
           total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGO_ZONE_DESC'], "")
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
           total_charge, distance, driver_pay, profit, rpm = get_city_aggregates(row['LEGD_ZONE_DESC'], "")
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
           title=f"Routes for Power Unit: {selected_punit}, Driver ID: {selected_driver}",
           geo=dict(scope="north america", projection_type="mercator"),
       )
       st.plotly_chart(fig)
       
       # Display locations missing coordinates relevant to the selection
       relevant_missing_origins = filtered_view[
           (pd.isna(filtered_view['ORIG_LAT']) | pd.isna(filtered_view['ORIG_LON']))
       ][['LEGO_ZONE_DESC']].drop_duplicates().rename(columns={
           'LEGO_ZONE_DESC': 'Missing Location'
       })
       
       relevant_missing_destinations = filtered_view[
           (pd.isna(filtered_view['DEST_LAT']) | pd.isna(filtered_view['DEST_LON']))
       ][['LEGD_ZONE_DESC']].drop_duplicates().rename(columns={
           'LEGD_ZONE_DESC': 'Missing Location'
       })
       
       relevant_missing_locations = pd.concat([relevant_missing_origins, relevant_missing_destinations]).drop_duplicates()
       
       if not relevant_missing_locations.empty:
           st.write("### Missing Locations")
           st.dataframe(relevant_missing_locations, use_container_width=True)

       else:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    
else:
    st.warning("Please upload both the TLORDER and DRIVERPAY CSV files to proceed.")











