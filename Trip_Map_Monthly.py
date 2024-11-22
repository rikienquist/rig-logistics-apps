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

    # Round Straight Distance to 1 decimal place
    route_summary_df['Straight Distance (miles)'] = route_summary_df['Straight Distance (miles)'].round(1)

    # Calculate grand totals
    total_charge = route_summary_df["Total Charge (CAD)"].sum()
    total_distance = route_summary_df["Distance (miles)"].sum()
    total_straight_distance = route_summary_df["Straight Distance (miles)"].sum()
    total_driver_pay = route_summary_df["Driver Pay (CAD)"].sum()
    total_profit = route_summary_df["Profit (CAD)"].sum()
    grand_revenue_per_mile = total_charge / total_distance if total_distance != 0 else 0

    # Add grand totals row
    grand_totals = {
        "Route": "Grand Totals",
        "BILL_NUMBER": "",
        "Total Charge (CAD)": f"${total_charge:,.2f}",
        "Distance (miles)": f"{total_distance:,.2f}",
        "Straight Distance (miles)": f"{total_straight_distance:,.1f}",
        "Revenue per Mile": f"${grand_revenue_per_mile:,.2f}",
        "Driver ID": "",
        "Driver Pay (CAD)": f"${total_driver_pay:,.2f}",
        "Profit (CAD)": f"${total_profit:,.2f}",
        "Date": ""
    }

    # Use pd.concat instead of append
    grand_totals_df = pd.DataFrame([grand_totals])  # Create a DataFrame for the totals
    route_summary_df = pd.concat([route_summary_df, grand_totals_df], ignore_index=True)

    # Format the DataFrame for display with currency formatting
    currency_columns = ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]
    for col in currency_columns:
        route_summary_df[col] = route_summary_df[col].apply(lambda x: f"${float(x):,.2f}" if isinstance(x, (int, float)) else x)

    # Display the table with grand totals
    st.write("Route Summary:")
    st.dataframe(route_summary_df, use_container_width=True)
