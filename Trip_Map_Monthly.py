import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from geopy.distance import geodesic
import os

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

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

# Load data from uploaded files or local paths
data_folder = "trip_map_data"
tlorder_df = pd.read_csv(os.path.join(data_folder, "TLORDER_Sep2022-Sep2024_V3.csv"), low_memory=False)
geocode_df = pd.read_csv(os.path.join(data_folder, "merged_geocoded.csv"), low_memory=False)
driver_pay_df = pd.read_csv(os.path.join(data_folder, "driver_pay_data.csv"), low_memory=False)

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

# Add Month Column for Grouping
filtered_df['PICK_UP_DATE'] = pd.to_datetime(filtered_df['PICK_UP_BY'])
filtered_df['Month'] = filtered_df['PICK_UP_DATE'].dt.to_period('M')

# Streamlit App
st.title("Trip Map Viewer by Month")

# PUNIT and Driver ID selection
punit_options = sorted(filtered_df['PICK_UP_PUNIT'].dropna().unique())
selected_punit = st.selectbox("Select PUNIT:", options=punit_options)

driver_options = ["All"] + sorted(filtered_df['DRIVER_ID'].dropna().astype(str))
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
    label_counter = 1
    for i, row in month_data.iterrows():
        # Plot origin and destination
        if label_counter == 1:  # Add legend for first instance only
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
        else:  # No legend for subsequent points
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON']],
                lat=[row['ORIG_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="blue"),
                text=str(label_counter),
                textposition="top right",
                showlegend=False,
            ))
            fig.add_trace(go.Scattergeo(
                lon=[row['DEST_LON']],
                lat=[row['DEST_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=str(label_counter + 1),
                textposition="top right",
                showlegend=False,
            ))
        # Plot line between origin and destination
        if label_counter == 1:  # Add legend for the first route only
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route",
            ))
        else:  # No legend for subsequent routes
            fig.add_trace(go.Scattergeo(
                lon=[row['ORIG_LON'], row['DEST_LON']],
                lat=[row['ORIG_LAT'], row['DEST_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                showlegend=False,
            ))
        label_counter += 2

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
            "Total Charge (CAD)": f"${row['TOTAL_CHARGE_CAD']:.2f}",
            "Distance (miles)": row['DISTANCE'],
            "Revenue per Mile": f"${row['Revenue per Mile']:.2f}",
            "Driver ID": row['DRIVER_ID'],
            "Driver Pay (CAD)": f"${row['TOTAL_PAY_AMT']:.2f}" if not pd.isna(row['TOTAL_PAY_AMT']) else "N/A",
            "Profit Margin (%)": f"{row['Profit Margin (%)']:.2f}%" if not pd.isna(row['Profit Margin (%)']) else "N/A",
            "Geopy_Distance": row['Geopy_Distance'],
            "Date": row['PICK_UP_DATE']
        })

    # Convert the route summary to a DataFrame
    route_summary_df = pd.DataFrame(route_summary)

    # Display the table
    st.write("Route Summary:")
    st.dataframe(route_summary_df)
else:
    st.warning("No data available for the selected PUNIT and Driver ID.")
