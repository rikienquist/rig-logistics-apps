import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import os

# Initialize global variables for navigation
if "day_index" not in st.session_state:
    st.session_state.day_index = 0

# Load city coordinates from the consolidated file
city_coordinates_file = "trip_map_data/city_coordinates.csv"
city_coordinates_df = pd.read_csv(city_coordinates_file)

# Load other datasets
data_folder = "trip_map_data"
tlorder_df = pd.read_csv(os.path.join(data_folder, "TLORDER_Sep2022-Sep2024_V3.csv"), low_memory=False)
driver_pay_df = pd.read_csv(os.path.join(data_folder, "driver_pay_data.csv"), low_memory=False)

# Preprocess city_coordinates_df for merging
city_coordinates_df.rename(columns={
    "city": "ORIGCITY",
    "province": "ORIGPROV",
    "latitude": "ORIG_LAT",
    "longitude": "ORIG_LON"
}, inplace=True)

# Merge origin coordinates
tlorder_df = tlorder_df.merge(city_coordinates_df, on=["ORIGCITY", "ORIGPROV"], how="left")

# Rename columns for destination and merge again
city_coordinates_df.rename(columns={
    "ORIGCITY": "DESTCITY",
    "ORIGPROV": "DESTPROV",
    "ORIG_LAT": "DEST_LAT",
    "ORIG_LON": "DEST_LON"
}, inplace=True)

tlorder_df = tlorder_df.merge(city_coordinates_df, on=["DESTCITY", "DESTPROV"], how="left")

# Filter for non-same-city routes and rows with valid coordinates
tlorder_df = tlorder_df[(tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) & 
                        (pd.notna(tlorder_df['ORIG_LAT'])) & 
                        (pd.notna(tlorder_df['DEST_LAT']))].copy()

# Merge with driver pay
driver_pay_agg = driver_pay_df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum', 'DRIVER_ID': 'first'})
tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

# Calculate CAD charge and filter
tlorder_df['TOTAL_CHARGE_CAD'] = tlorder_df.apply(
    lambda x: (x['CHARGES'] + x['XCHARGES']) * 1.38 if x['CURRENCY_CODE'] == 'USD' else x['CHARGES'] + x['XCHARGES'], 
    axis=1
)
filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & (tlorder_df['DISTANCE'] != 0)].copy()

# Ensure PICK_UP_PUNIT is clean
filtered_df['PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].astype(str).fillna("Unknown")

# Calculate Revenue per Mile and Profit Margin
filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df['Profit Margin (%)'] = (filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['TOTAL_PAY_AMT']) * 100

# Add Date column for grouping
filtered_df['PICK_UP_DATE'] = pd.to_datetime(filtered_df['PICK_UP_BY'])

# Calculate Straight Distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Radius of Earth in miles
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

filtered_df['Straight Distance'] = haversine(
    filtered_df['ORIG_LAT'], filtered_df['ORIG_LON'],
    filtered_df['DEST_LAT'], filtered_df['DEST_LON']
)

# Streamlit App
st.title("Trip Map Viewer by Day")

# PUNIT and Driver ID selection
punit_options = sorted(filtered_df['PICK_UP_PUNIT'].dropna().unique())
selected_punit = st.selectbox("Select PUNIT:", options=punit_options)

# Filter Driver IDs based on selected PUNIT
relevant_drivers = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit]['DRIVER_ID'].dropna().unique()
driver_options = ["All"] + sorted(relevant_drivers.astype(str))
selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)

# Filter based on selections
filtered_view = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit].copy()
if selected_driver != "All":
    filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()

# Day navigation
days = sorted(filtered_view['PICK_UP_DATE'].dropna().unique())
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
    day_data = filtered_view[filtered_view['PICK_UP_DATE'] == selected_day].copy()

    # Generate map
    fig = go.Figure()
    label_counter = 1
    legend_added = {"Origin": False, "Destination": False, "Route": False}  # Track if the legend was added

    for _, row in day_data.iterrows():
        # Add Origin point
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON']],
            lat=[row['ORIG_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="blue"),
            text=str(label_counter),
            textposition="top right",
            name="Origin" if not legend_added["Origin"] else None,
            hovertext=(f"City: {row['ORIGCITY']}, {row['ORIGPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Distance (miles): {row['DISTANCE']}<br>"
                       f"Straight Distance (miles): {row['Straight Distance']:.2f}"),
            hoverinfo="text",
            showlegend=not legend_added["Origin"],
        ))
        legend_added["Origin"] = True

        # Add Destination point
        fig.add_trace(go.Scattergeo(
            lon=[row['DEST_LON']],
            lat=[row['DEST_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=str(label_counter + 1),
            textposition="top right",
            name="Destination" if not legend_added["Destination"] else None,
            hovertext=(f"City: {row['DESTCITY']}, {row['DESTPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Distance (miles): {row['DISTANCE']}<br>"
                       f"Straight Distance (miles): {row['Straight Distance']:.2f}"),
            hoverinfo="text",
            showlegend=not legend_added["Destination"],
        ))
        legend_added["Destination"] = True

        # Add Route line
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Route" if not legend_added["Route"] else None,
            hoverinfo="skip",
            showlegend=not legend_added["Route"],
        ))
        legend_added["Route"] = True

        label_counter += 2

    fig.update_layout(
        title=f"Routes for {selected_day.date()} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
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
            "Straight Distance (miles)": row['Straight Distance'],
            "Revenue per Mile": f"${row['Revenue per Mile']:.2f}",
            "Driver ID": row['DRIVER_ID'],
            "Driver Pay (CAD)": f"${row['TOTAL_PAY_AMT']:.2f}" if not pd.isna(row['TOTAL_PAY_AMT']) else "N/A",
            "Profit Margin (%)": f"{row['Profit Margin (%)']:.2f}%" if not pd.isna(row['Profit Margin (%)']) else "N/A",
            "Date": row['PICK_UP_DATE']
        })

    # Convert the route summary to a DataFrame
    route_summary_df = pd.DataFrame(route_summary)

    # Highlight same-day routes (only one day here, so no effect)
    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)

else:
    st.warning("No data available for the selected PUNIT and Driver ID.")
