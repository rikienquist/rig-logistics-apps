import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import os

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Load the city coordinates from the consolidated file
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

# Calculate Revenue per Mile and Profit
filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df['Profit'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_AMT']

# Add Month Column for Grouping
filtered_df['PICK_UP_DATE'] = pd.to_datetime(filtered_df['PICK_UP_BY'])
filtered_df['Month'] = filtered_df['PICK_UP_DATE'].dt.to_period('M')

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
st.title("Trip Map Viewer by Month")

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

# Month navigation
months = sorted(filtered_view['Month'].dropna().unique())
total_months = len(months)

def navigate_months(direction):
    if direction == "previous" and st.session_state.month_index > 0:
        st.session_state.month_index -= 1
    elif direction == "next" and st.session_state.month_index < total_months - 1:
        st.session_state.month_index += 1
    elif direction == "back_12" and st.session_state.month_index > 11:
        st.session_state.month_index -= 12
    elif direction == "ahead_12" and st.session_state.month_index < total_months - 12:
        st.session_state.month_index += 12

col1, col2, col3, col4 = st.columns(4)
col1.button("Previous Month", on_click=navigate_months, args=("previous",))
col2.button("Next Month", on_click=navigate_months, args=("next",))
col3.button("Back 12 Months", on_click=navigate_months, args=("back_12",))
col4.button("Ahead 12 Months", on_click=navigate_months, args=("ahead_12",))

if total_months > 0:
    selected_month = months[st.session_state.month_index]
    st.write(f"Viewing data for month: {selected_month}")
    month_data = filtered_view[filtered_view['Month'] == selected_month].copy()

    # Generate map
    fig = go.Figure()
    month_data = month_data.sort_values(by='PICK_UP_DATE')  # Sort routes chronologically

    # Add map traces
    label_counter = 1
    legend_added = {"Origin": False, "Destination": False, "Route": False}
    for _, row in month_data.iterrows():
        # Origin marker
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON']],
            lat=[row['ORIG_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="blue"),
            text=str(label_counter),
            textposition="top right",
            name="Origin" if not legend_added["Origin"] else None,
            hoverinfo="text",
            hovertext=(f"City: {row['ORIGCITY']}, {row['ORIGPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Straight Distance (miles): {row['Straight Distance']:.1f}"),
            showlegend=not legend_added["Origin"],
        ))
        legend_added["Origin"] = True

        # Destination marker
        fig.add_trace(go.Scattergeo(
            lon=[row['DEST_LON']],
            lat=[row['DEST_LAT']],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=str(label_counter + 1),
            textposition="top right",
            name="Destination" if not legend_added["Destination"] else None,
            hoverinfo="text",
            hovertext=(f"City: {row['DESTCITY']}, {row['DESTPROV']}<br>"
                       f"Date: {row['PICK_UP_DATE']}<br>"
                       f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                       f"Straight Distance (miles): {row['Straight Distance']:.1f}"),
            showlegend=not legend_added["Destination"],
        ))
        legend_added["Destination"] = True

        # Route line
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

    # Update map layout
    fig.update_layout(
        title=f"Routes for {selected_month} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
        geo=dict(scope="north america", projection_type="mercator"),
    )
    st.plotly_chart(fig)

    # Create the route summary table
    route_summary = []
    for _, row in month_data.iterrows():
        route_summary.append({
            "Route": f"{row['ORIGCITY']}, {row['ORIGPROV']} to {row['DESTCITY']}, {row['DESTPROV']}",
            "BILL_NUMBER": row['BILL_NUMBER'],
            "Total Charge (CAD)": row['TOTAL_CHARGE_CAD'],
            "Distance (miles)": row['DISTANCE'],
            "Straight Distance (miles)": row['Straight Distance'],
            "Revenue per Mile": row['Revenue per Mile'],
            "Driver ID": row['DRIVER_ID'],
            "Driver Pay (CAD)": row['TOTAL_PAY_AMT'],
            "Profit (CAD)": row['Profit'],
            "Date": row['PICK_UP_DATE']
        })
    
    # Convert the route summary to a DataFrame
    route_summary_df = pd.DataFrame(route_summary)

    # Format the DataFrame for display
    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)

else:
    st.warning("No data available for the selected PUNIT and Driver ID.")
