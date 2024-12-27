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
uploaded_tlorder_driverpay_file = st.file_uploader("Upload TLORDER + DRIVERPAY CSV file", type="csv")

@st.cache_data
def load_city_coordinates():
    """Load location coordinates from a fixed file."""
    city_coords = pd.read_csv("trip_map_data/location_coordinates.csv")
    city_coords.rename(columns={
        "location": "LOCATION",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)
    return city_coords

@st.cache_data
def preprocess_legsum(file):
    """Preprocess LEGSUM data."""
    df = pd.read_csv(file, low_memory=False, parse_dates=["LS_ACTUAL_DATE"])
    return df

@st.cache_data
def preprocess_tlorder_driverpay(file):
    """Preprocess TLORDER + DRIVERPAY data."""
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_SUM'] = pd.to_numeric(df['TOTAL_PAY_SUM'], errors='coerce')
    return df

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    legsum_df = preprocess_legsum(uploaded_legsum_file)
    tlorder_driverpay_df = preprocess_tlorder_driverpay(uploaded_tlorder_driverpay_file)

    # Merge LEGSUM and TLORDER+DRIVERPAY
    merged_df = legsum_df.merge(tlorder_driverpay_df, left_on='LS_FREIGHT', right_on='BILL_NUMBER', how='left')

    # User date range input
    st.sidebar.header("Select Date Range")
    start_date = st.sidebar.date_input("Start Date", value=merged_df['LS_ACTUAL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=merged_df['LS_ACTUAL_DATE'].max())

    # Filter by date range
    filtered_df = merged_df[
        (merged_df['LS_ACTUAL_DATE'] >= pd.to_datetime(start_date)) &
        (merged_df['LS_ACTUAL_DATE'] <= pd.to_datetime(end_date))
    ].copy()

    # Merge coordinates for LEGO_ZONE_DESC and LEGD_ZONE_DESC
    city_coordinates_df.rename(columns={"LOCATION": "LEGO_ZONE_DESC"}, inplace=True)
    filtered_df = filtered_df.merge(city_coordinates_df, on="LEGO_ZONE_DESC", how="left", suffixes=("", "_ORIG"))
    city_coordinates_df.rename(columns={"LEGO_ZONE_DESC": "LEGD_ZONE_DESC"}, inplace=True)
    filtered_df = filtered_df.merge(city_coordinates_df, on="LEGD_ZONE_DESC", how="left", suffixes=("", "_DEST"))

    # Calculate straight distance
    def calculate_haversine(lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on Earth."""
        R = 3958.8  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    filtered_df['Straight Distance (miles)'] = calculate_haversine(
        filtered_df['LAT'], filtered_df['LON'],
        filtered_df['LAT_DEST'], filtered_df['LON_DEST']
    )

    # Calculate revenue per mile and profit
    filtered_df['Revenue per Mile'] = filtered_df['CHARGES'] / filtered_df['LS_LEG_DIST']
    filtered_df['Profit (CAD)'] = filtered_df['CHARGES'] - filtered_df['TOTAL_PAY_SUM']

    # Route Summary Table
    route_summary = filtered_df[[
        'LEGO_ZONE_DESC', 'LEGD_ZONE_DESC', 'BILL_NUMBER', 'LS_TRIP_NUMBER',
        'LS_LEG_DIST', 'LS_MT_LOADED', 'CHARGES', 'Straight Distance (miles)',
        'Revenue per Mile', 'LS_ACTUAL_DATE', 'LS_LEG_NOTE'
    ]]
    route_summary.rename(columns={
        'LEGO_ZONE_DESC': 'Origin', 
        'LEGD_ZONE_DESC': 'Destination',
        'CHARGES': 'Total Charge (CAD)', 
        'LS_ACTUAL_DATE': 'Date'
    }, inplace=True)
    st.write("Route Summary")
    st.dataframe(route_summary, use_container_width=True)

    # Missing Locations
    missing_locations = pd.concat([
        filtered_df[pd.isna(filtered_df['LAT']) | pd.isna(filtered_df['LON'])][['LEGO_ZONE_DESC']].rename(columns={'LEGO_ZONE_DESC': 'Location'}),
        filtered_df[pd.isna(filtered_df['LAT_DEST']) | pd.isna(filtered_df['LON_DEST'])][['LEGD_ZONE_DESC']].rename(columns={'LEGD_ZONE_DESC': 'Location'})
    ]).drop_duplicates()

    if not missing_locations.empty:
        st.write("### Missing Locations")
        st.dataframe(missing_locations, use_container_width=True)

    # Map Visualization
    fig = go.Figure()

    for _, row in filtered_df.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row['LON'], row['LON_DEST']],
            lat=[row['LAT'], row['LAT_DEST']],
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(width=2),
            hovertext=(
                f"Route: {row['LEGO_ZONE_DESC']} to {row['LEGD_ZONE_DESC']}<br>"
                f"Total Charge (CAD): {row['CHARGES']:.2f}<br>"
                f"Revenue per Mile: {row['Revenue per Mile']:.2f}<br>"
                f"Profit: {row['Profit (CAD)']:.2f}"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=f"Route Map ({start_date} to {end_date})",
        geo=dict(scope='north america', projection_type='mercator')
    )
    st.plotly_chart(fig)

else:
    st.warning("Please upload both LEGSUM and TLORDER+DRIVERPAY CSV files.")

# Helper function for aggregation
def aggregate_city_data(filtered_df, city_coordinates_df):
    """Aggregate data by origin and destination locations."""
    city_data = pd.concat([
        filtered_df[['LEGO_ZONE_DESC', 'CHARGES', 'LS_LEG_DIST', 'TOTAL_PAY_SUM']].rename(
            columns={'LEGO_ZONE_DESC': 'Location', 'CHARGES': 'Total Charge (CAD)',
                     'LS_LEG_DIST': 'Distance (miles)', 'TOTAL_PAY_SUM': 'Driver Pay (CAD)'}),
        filtered_df[['LEGD_ZONE_DESC', 'CHARGES', 'LS_LEG_DIST', 'TOTAL_PAY_SUM']].rename(
            columns={'LEGD_ZONE_DESC': 'Location', 'CHARGES': 'Total Charge (CAD)',
                     'LS_LEG_DIST': 'Distance (miles)', 'TOTAL_PAY_SUM': 'Driver Pay (CAD)'})
    ])
    city_data = city_data.groupby('Location', as_index=False).sum()
    city_data = city_data.merge(city_coordinates_df, left_on='Location', right_on='LOCATION', how='left')

    # Calculate Revenue per Mile and Profit
    city_data['Revenue per Mile'] = city_data['Total Charge (CAD)'] / city_data['Distance (miles)']
    city_data['Profit (CAD)'] = city_data['Total Charge (CAD)'] - city_data['Driver Pay (CAD)']
    return city_data

# Aggregate city data
if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    city_aggregates = aggregate_city_data(filtered_df, city_coordinates_df)

    # Display aggregated city data
    st.write("### City Data Aggregates")
    st.dataframe(city_aggregates, use_container_width=True)

    # Create aggregated map visualization
    fig_agg = go.Figure()
    for _, row in city_aggregates.iterrows():
        fig_agg.add_trace(go.Scattergeo(
            lon=[row['LON']],
            lat=[row['LAT']],
            mode="markers",
            marker=dict(size=10),
            hovertext=(
                f"Location: {row['Location']}<br>"
                f"Total Charge (CAD): {row['Total Charge (CAD)']:.2f}<br>"
                f"Distance (miles): {row['Distance (miles)']:.2f}<br>"
                f"Revenue per Mile: {row['Revenue per Mile']:.2f}<br>"
                f"Driver Pay (CAD): {row['Driver Pay (CAD)']:.2f}<br>"
                f"Profit (CAD): {row['Profit (CAD)']:.2f}"
            ),
            showlegend=False
        ))

    fig_agg.update_layout(
        title="City Data Aggregates Map",
        geo=dict(scope='north america', projection_type='mercator')
    )
    st.plotly_chart(fig_agg)

# Create a grand total summary
def generate_grand_totals(route_summary):
    """Calculate grand totals for the route summary."""
    total_charge = route_summary['Total Charge (CAD)'].sum()
    total_distance = route_summary['LS_LEG_DIST'].sum()
    total_driver_pay = route_summary['Driver Pay (CAD)'].sum()
    total_profit = route_summary['Profit (CAD)'].sum()

    revenue_per_mile = total_charge / total_distance if total_distance > 0 else 0

    grand_totals = pd.DataFrame([{
        "Route": "Grand Totals",
        "Total Charge (CAD)": total_charge,
        "Distance (miles)": total_distance,
        "Driver Pay (CAD)": total_driver_pay,
        "Profit (CAD)": total_profit,
        "Revenue per Mile": revenue_per_mile
    }])

    return grand_totals

if uploaded_legsum_file and uploaded_tlorder_driverpay_file:
    # Combine route summary with grand totals
    grand_totals = generate_grand_totals(route_summary)
    full_summary = pd.concat([route_summary, grand_totals], ignore_index=True)

    # Display formatted grand totals
    st.write("### Route Summary with Grand Totals")
    st.dataframe(full_summary, use_container_width=True)

    # Apply formatting for better visualization
    full_summary['Total Charge (CAD)'] = full_summary['Total Charge (CAD)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    full_summary['Revenue per Mile'] = full_summary['Revenue per Mile'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    full_summary['Driver Pay (CAD)'] = full_summary['Driver Pay (CAD)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    full_summary['Profit (CAD)'] = full_summary['Profit (CAD)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    full_summary['Distance (miles)'] = full_summary['Distance (miles)'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "")

    st.dataframe(full_summary.style.set_precision(2), use_container_width=True)

    # Grand totals as a summary section
    st.markdown(f"""
    ### Grand Totals:
    - **Total Charge (CAD):** ${grand_totals['Total Charge (CAD)'].iloc[0]:,.2f}
    - **Total Distance (miles):** {grand_totals['Distance (miles)'].iloc[0]:,.1f}
    - **Total Driver Pay (CAD):** ${grand_totals['Driver Pay (CAD)'].iloc[0]:,.2f}
    - **Total Profit (CAD):** ${grand_totals['Profit (CAD)'].iloc[0]:,.2f}
    - **Revenue per Mile:** ${grand_totals['Revenue per Mile'].iloc[0]:,.2f}
    """)

else:
    st.warning("Please upload both LEGSUM and TLORDER+DRIVERPAY CSV files to proceed.")

