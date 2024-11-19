import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from geopy.distance import geodesic
import os

# Initialize global variables for navigation
if "day_index" not in st.session_state:
    st.session_state.day_index = 0

# Dictionary for city-province coordinate corrections
coordinate_fixes = {
    ("ACHESON", "AB"): {"LAT": 53.5522, "LON": -113.7627},
    ("BALZAC", "AB"): {"LAT": 51.2126, "LON": -114.0076},
    ("LACOMBE", "AB"): {"LAT": 54.4682, "LON": -113.7305},
    ("RED WATER", "AB"): {"LAT": 53.9520, "LON": -113.1089},
    ("BROOKS", "AB"): {"LAT": 50.5657, "LON": -111.8978},
    ("Hardisty", "AB"): {"LAT": 52.6734, "LON": -111.3075},
    ("Falun", "AB"): {"LAT": 52.9598, "LON": -113.8253},
    ("STETTLER", "AB"): {"LAT": 52.3235, "LON": -112.7192},
    ("MANVILLE", "AB"): {"LAT": 53.3402, "LON": -111.1752},
    ("Thorsby", "AB"): {"LAT": 53.2272, "LON": -114.0480},
    ("BEAUMONT", "AB"): {"LAT": 53.3521, "LON": -113.4151},
    ("BOYLE", "AB"): {"LAT": 54.5873, "LON": -112.8034},
    ("GRAND CACHE", "AB"): {"LAT": 53.8886, "LON": -119.1143},
    ("MILLET", "AB"): {"LAT": 53.0907, "LON": -113.4685},
    ("NELSON", "BC"): {"LAT": 49.4928, "LON": -117.2948},
    ("SURREY", "BC"): {"LAT": 49.1913, "LON": -122.8490},
    ("DELTA", "BC"): {"LAT": 49.0952, "LON": -123.0265},
    ("GRAND FORKS", "BC"): {"LAT": 49.0335, "LON": -118.4389},
    ("CRESTON", "BC"): {"LAT": 49.0955, "LON": -116.5135},
    ("RICHMOND", "BC"): {"LAT": 49.1666, "LON": -123.1336},
    ("BLUMENORT", "MB"): {"LAT": 49.6079, "LON": -96.6935},
    ("ST ANDREWS", "MB"): {"LAT": 50.2625, "LON": -96.9842},
    ("MORDEN", "MB"): {"LAT": 49.1923, "LON": -98.1143},
    ("STEINBACH", "MB"): {"LAT": 49.5303, "LON": -96.6912},
    ("WINKLER", "MB"): {"LAT": 49.1802, "LON": -97.9391},
    ("BOLTON", "ON"): {"LAT": 43.8742, "LON": -79.7307},
    ("LIVELY", "ON"): {"LAT": 46.4366, "LON": -81.1466},
    ("VERNON", "CA"): {"LAT": 34.0039, "LON": -118.2301},
    ("SELMA", "CA"): {"LAT": 36.5708, "LON": -119.6121},
    ("TROY", "NY"): {"LAT": 42.7284, "LON": -73.6918},
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

# Merge geocodes
tlorder_df = tlorder_df.merge(
    geocode_df[['ORIGCITY', 'ORIG_LAT', 'ORIG_LON']].drop_duplicates(),
    on='ORIGCITY', how='left'
).merge(
    geocode_df[['DESTCITY', 'DEST_LAT', 'DEST_LON']].drop_duplicates(),
    on='DESTCITY', how='left'
)
tlorder_df = tlorder_df.apply(correct_coordinates, axis=1)

# Filter for non-same-city routes
tlorder_df = tlorder_df[tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']].copy()

# Merge with driver pay
driver_pay_agg = driver_pay_df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum', 'DRIVER_ID': 'first'})
tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

# Calculate CAD charge and filter
tlorder_df['TOTAL_CHARGE_CAD'] = tlorder_df.apply(
    lambda x: (x['CHARGES'] + x['XCHARGES']) * 1.38 if x['CURRENCY_CODE'] == 'USD' else x['CHARGES'] + x['XCHARGES'], 
    axis=1
)
filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & (tlorder_df['DISTANCE'] != 0)].copy()
filtered_df.dropna(subset=['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON'], inplace=True)

# Ensure PICK_UP_PUNIT is clean
filtered_df.loc[:, 'PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].astype(str).fillna("Unknown")

# Calculate Revenue per Mile and Profit Margin
filtered_df.loc[:, 'Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df.loc[:, 'Profit Margin (%)'] = (filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['TOTAL_PAY_AMT']) * 100

# Calculate Geopy Distance
filtered_df['route_key'] = filtered_df[['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON']].apply(tuple, axis=1)
unique_routes = filtered_df.drop_duplicates(subset='route_key').copy()
unique_routes['Geopy_Distance'] = unique_routes['route_key'].apply(
    lambda r: geodesic((r[0], r[1]), (r[2], r[3])).miles
)
filtered_df = filtered_df.merge(unique_routes[['route_key', 'Geopy_Distance']], on='route_key', how='left')

# Trip classification logic
def classify_trips(group):
    unique_geopy_distance_sum = group.drop_duplicates(['ORIGCITY', 'DESTCITY'])['Geopy_Distance'].sum()
    total_distance = group['DISTANCE'].sum()

    if len(group) == 1:
        # Single BILL_NUMBER logic
        row = group.iloc[0]
        if row['DISTANCE'] < 2 * row['Geopy_Distance']:
            row['Trip Type'] = 'One-Way'
            row['Round-Trip Distance'] = row['DISTANCE'] * 2
            row['One-Way Distance'] = row['DISTANCE']
        else:
            row['Trip Type'] = 'Round-Trip'
            row['One-Way Distance'] = row['DISTANCE'] / 2
            row['Round-Trip Distance'] = row['DISTANCE']
        return pd.DataFrame([row])

    if total_distance < 2 * unique_geopy_distance_sum:
        # Each BILL_NUMBER is a one-way
        group['Trip Type'] = 'One-Way'
        group['One-Way Distance'] = group['DISTANCE']  # Each row's distance is its own One-Way Distance
        group['Round-Trip Distance'] = total_distance  # Total distance is the Round-Trip Distance for all rows
    else:
        # Each BILL_NUMBER is part of a round-trip
        group['Trip Type'] = 'Round-Trip'
        group['Round-Trip Distance'] = total_distance  # Total distance for all
        group['One-Way Distance'] = total_distance / 2  # Half the total distance as One-Way Distance
    return group

# Apply trip classification to filtered_df
filtered_df = filtered_df.groupby(['PICK_UP_PUNIT', 'PICK_UP_BY'], group_keys=False).apply(classify_trips).reset_index(drop=True)

# Streamlit App
st.title("Trip Map Viewer")

# PUNIT and Driver ID selection
punit_options = sorted(filtered_df['PICK_UP_PUNIT'].dropna().unique())
selected_punit = st.selectbox("Select PUNIT:", options=punit_options)

driver_options = ["All"] + sorted(filtered_df['DRIVER_ID'].dropna().astype(str))
selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)

# Filter based on selections
filtered_view = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit].copy()
if selected_driver != "All":
    filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()

# Day navigation
days = sorted(filtered_view['PICK_UP_BY'].dropna().unique())
total_days = len(days)

def navigate_days(direction):
    if direction == "previous" and st.session_state.day_index > 0:
        st.session_state.day_index -= 1
    elif direction == "next" and st.session_state.day_index < total_days - 1:
        st.session_state.day_index += 1
    elif direction == "back_50" and st.session_state.day_index > 49:
        st.session_state.day_index -= 50
    elif direction == "ahead_50" and st.session_state.day_index < total_days - 50:
        st.session_state.day_index += 50

col1, col2, col3, col4 = st.columns(4)
col1.button("Previous Day", on_click=navigate_days, args=("previous",))
col2.button("Next Day", on_click=navigate_days, args=("next",))
col3.button("Back 50 Days", on_click=navigate_days, args=("back_50",))
col4.button("Ahead 50 Days", on_click=navigate_days, args=("ahead_50",))

if total_days > 0:
    selected_day = days[st.session_state.day_index]
    st.write(f"Viewing data for day: {selected_day}")
    day_data = filtered_view[filtered_view['PICK_UP_BY'] == selected_day].copy()

    # Generate map
    fig = go.Figure()
    label_counter = 1
    for _, row in day_data.iterrows():
        # Plot origin and destination
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON']],
            lat=[row['ORIG_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="blue"),
            text=str(label_counter),
            textposition="top right",
            name="Origin",
        ))
        fig.add_trace(go.Scattergeo(
            lon=[row['DEST_LON']],
            lat=[row['DEST_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=str(label_counter + 1),
            textposition="top right",
            name="Destination",
        ))
        # Plot line between origin and destination
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Route",
        ))
        label_counter += 2

    fig.update_layout(
        title=f"Routes for {selected_day} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
        geo=dict(scope="north america", projection_type="mercator"),
    )
    st.plotly_chart(fig)

    # Create the route summary table
    route_summary = []
    for _, row in day_data.iterrows():
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

    # Convert the route summary to a DataFrame
    route_summary_df = pd.DataFrame(route_summary)

    # Display the table
    st.write("Route Summary:")
    st.dataframe(route_summary_df)

else:
    st.warning("No data available for the selected PUNIT and Driver ID.")
