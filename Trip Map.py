import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import geodesic
import os

# Set paths
data_folder = "trip_map_data"
tlorder_path = os.path.join(data_folder, "TLORDER_Sep2022-Sep2024_V3.csv")
geocode_path = os.path.join(data_folder, "merged_geocoded.csv")
driver_pay_path = os.path.join(data_folder, "driver_pay_data.csv")

# Load the primary datasets
tlorder_df = pd.read_csv(tlorder_path, low_memory=False)
geocode_df = pd.read_csv(geocode_path, low_memory=False)
driver_pay_df = pd.read_csv(driver_pay_path, low_memory=False)

# Dictionary for city-province coordinate corrections
coordinate_fixes = {
    ("ACHESON", "AB"): {"LAT": 53.5522, "LON": -113.7627},
    ("BALZAC", "AB"): {"LAT": 51.2126, "LON": -114.0076},
    # [Truncated for brevity]
    ("MOTLEY", "MN"): {"LAT": 46.3366, "LON": -94.6462},
}

# Function to apply coordinate corrections
def correct_coordinates(row):
    orig_key = (row['ORIGCITY'], row['ORIGPROV'])
    dest_key = (row['DESTCITY'], row['DESTPROV'])
    if orig_key in coordinate_fixes:
        row['ORIG_LAT'], row['ORIG_LON'] = coordinate_fixes[orig_key].values()
    if dest_key in coordinate_fixes:
        row['DEST_LAT'], row['DEST_LON'] = coordinate_fixes[dest_key].values()
    return row

# Merge coordinates and apply corrections
tlorder_df = tlorder_df.merge(
    geocode_df[['ORIGCITY', 'ORIG_LAT', 'ORIG_LON']].drop_duplicates(),
    on='ORIGCITY', how='left'
).merge(
    geocode_df[['DESTCITY', 'DEST_LAT', 'DEST_LON']].drop_duplicates(),
    on='DESTCITY', how='left'
).apply(correct_coordinates, axis=1)

# Exclude same-city routes
tlorder_df = tlorder_df[tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']]

# Merge driver pay
driver_pay_agg = driver_pay_df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum', 'DRIVER_ID': 'first'})
tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

# Calculate CAD charge and filter
tlorder_df['TOTAL_CHARGE_CAD'] = tlorder_df.apply(
    lambda x: (x['CHARGES'] + x['XCHARGES']) * 1.38 if x['CURRENCY_CODE'] == 'USD' else x['CHARGES'] + x['XCHARGES'], 
    axis=1
)
filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & (tlorder_df['DISTANCE'] != 0)]
filtered_df.dropna(subset=['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON'], inplace=True)

# Calculate Revenue per Mile and Profit Margin
filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df['Profit Margin (%)'] = (filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['TOTAL_PAY_AMT']) * 100

# Calculate Geopy Distance only for unique routes
filtered_df['route_key'] = filtered_df[['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON']].apply(tuple, axis=1)
unique_routes = filtered_df.drop_duplicates(subset='route_key')
unique_routes['Geopy_Distance'] = unique_routes['route_key'].apply(
    lambda r: geodesic((r[0], r[1]), (r[2], r[3])).miles
)
filtered_df = filtered_df.merge(unique_routes[['route_key', 'Geopy_Distance']], on='route_key', how='left')

# Streamlit app layout
st.title("Trip Route Analysis Dashboard")

# Sidebar filters
selected_punit = st.sidebar.selectbox("Select PUNIT", sorted(filtered_df['PICK_UP_PUNIT'].astype(str).unique()))
driver_options = sorted(filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit]['DRIVER_ID'].unique())
selected_driver = st.sidebar.selectbox("Select Driver ID (Optional)", driver_options, index=0)

# Filter based on selections
data = filtered_df
if selected_punit:
    data = data[data['PICK_UP_PUNIT'] == selected_punit]
if selected_driver:
    data = data[data['DRIVER_ID'] == selected_driver]

# Route visualization
st.subheader(f"Routes for PUNIT: {selected_punit} and Driver ID: {selected_driver or 'All'}")
day_routes = data.groupby(data['PICK_UP_BY'])

day_index = st.sidebar.slider("Select Day", 0, len(day_routes) - 1, 0)
day, day_data = list(day_routes)[day_index]

route_summary = []

# Create the map with Plotly
fig = go.Figure()

for _, row in day_data.iterrows():
    # Route summary
    route_summary.append({
        "Route": f"{row['ORIGCITY']}, {row['ORIGPROV']} to {row['DESTCITY']}, {row['DESTPROV']}",
        "BILL_NUMBER": row['BILL_NUMBER'],
        "Total Charge (CAD)": f"${row['TOTAL_CHARGE_CAD']:.2f}",
        "Distance (miles)": row['DISTANCE'],
        "Revenue per Mile": f"${row['Revenue per Mile']:.2f}",
        "Driver ID": row['DRIVER_ID'],
        "Driver Pay (CAD)": f"${row['TOTAL_PAY_AMT']:.2f}" if not pd.isna(row['TOTAL_PAY_AMT']) else "N/A",
        "Profit Margin (%)": f"{row['Profit Margin (%)']:.2f}%" if not pd.isna(row['Profit Margin (%)']) else "N/A",
        "Geopy_Distance": row['Geopy_Distance'],
        "One-Way Distance": row['One-Way Distance'],
        "Round-Trip Distance": row['Round-Trip Distance'],
        "Trip Type": row['Trip Type'],
        "Date": row['PICK_UP_BY']
    })

    fig.add_trace(go.Scattergeo(
        lon=[row['ORIG_LON'], row['DEST_LON']],
        lat=[row['ORIG_LAT'], row['DEST_LAT']],
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=8, color='red'),
        hoverinfo="text",
        name=f"{row['ORIGCITY']} to {row['DESTCITY']}",
        text=f"{row['ORIGCITY']} to {row['DESTCITY']}<br>Distance: {row['DISTANCE']} miles"
    ))

# Update map layout
fig.update_layout(
    title=f"Routes for {day} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
    geo=dict(scope='north america', projection_type='mercator'),
)

# Display map and route summary
st.plotly_chart(fig)
st.write("Route Summary", pd.DataFrame(route_summary))

