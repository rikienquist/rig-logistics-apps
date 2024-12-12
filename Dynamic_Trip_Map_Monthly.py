import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer by Month")

st.markdown("""
### Instructions:
Use the following query to generate the required TLORDER data:  
`SELECT BILL_NUMBER, DETAIL_LINE_ID, CALLNAME, ORIGCITY, ORIGPROV, DESTCITY, DESTPROV, PICK_UP_PUNIT, DELIVERY_PUNIT, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE, PICK_UP_BY, DELIVER_BY  
FROM TLORDER WHERE "PICK_UP_BY" BETWEEN 'X' AND 'Y';`  

Use the following query to generate the required DRIVERPAY data:  
`SELECT BILL_NUMBER, PAY_ID, DRIVER_ID, PAY_DESCRIPTION, FB_TOTAL_CHARGES, CURRENCY_CODE, TOTAL_PAY_AMT, PAID_DATE, DATE_TRANS  
FROM DRIVERPAY WHERE "PAID_DATE" BETWEEN 'X' AND 'Y';`  

Replace X and Y with the desired date range in form YYYY-MM-DD.  

Save the query results as CSV files and upload them below to visualize the data.
""")

# File upload section
uploaded_tlorder_file = st.file_uploader("Upload TLORDER CSV file", type="csv")
uploaded_driverpay_file = st.file_uploader("Upload DRIVERPAY CSV file", type="csv")

@st.cache_data
def load_city_coordinates():
    city_coords = pd.read_csv("trip_map_data/city_coordinates.csv")
    city_coords.rename(columns={
        "city": "CITY",
        "province": "PROVINCE",
        "latitude": "LAT",
        "longitude": "LON"
    }, inplace=True)
    return city_coords

@st.cache_data
def preprocess_tlorder(file, city_coords):
    df = pd.read_csv(file, low_memory=False)
    origin_coords = city_coords.rename(columns={"CITY": "ORIGCITY", "PROVINCE": "ORIGPROV", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(origin_coords, on=["ORIGCITY", "ORIGPROV"], how="left")
    dest_coords = city_coords.rename(columns={"CITY": "DESTCITY", "PROVINCE": "DESTPROV", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    df = df.merge(dest_coords, on=["DESTCITY", "DESTPROV"], how="left")
    return df

@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_AMT'] = pd.to_numeric(df['TOTAL_PAY_AMT'], errors='coerce')
    driver_pay_agg = df.groupby('BILL_NUMBER').agg({
        'TOTAL_PAY_AMT': 'sum',
        'DRIVER_ID': 'first'
    }).reset_index()
    return driver_pay_agg

@st.cache_data
def calculate_haversine(df):
    R = 3958.8
    lat1, lon1 = np.radians(df['ORIG_LAT']), np.radians(df['ORIG_LON'])
    lat2, lon2 = np.radians(df['DEST_LAT']), np.radians(df['DEST_LON'])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

if uploaded_tlorder_file and uploaded_driverpay_file:
    city_coordinates_df = load_city_coordinates()
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file, city_coordinates_df)
    driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)

    tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

    tlorder_df = tlorder_df[
        (tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) &
        (pd.notna(tlorder_df['ORIG_LAT'])) &
        (pd.notna(tlorder_df['DEST_LAT']))
    ].copy()

    exchange_rate = 1.38
    tlorder_df['CHARGES'] = pd.to_numeric(tlorder_df['CHARGES'], errors='coerce')
    tlorder_df['XCHARGES'] = pd.to_numeric(tlorder_df['XCHARGES'], errors='coerce')
    tlorder_df['TOTAL_CHARGE_CAD'] = np.where(
        tlorder_df['CURRENCY_CODE'] == 'USD',
        (tlorder_df['CHARGES'] + tlorder_df['XCHARGES']) * exchange_rate,
        tlorder_df['CHARGES'] + tlorder_df['XCHARGES']
    )

    filtered_df = tlorder_df[
        (tlorder_df['TOTAL_CHARGE_CAD'] != 0) & 
        (tlorder_df['DISTANCE'] != 0)
    ].copy()

    filtered_df['PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].fillna("Unknown").astype(str)
    filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
    filtered_df['Profit (CAD)'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_AMT']

    cutoff_date = pd.Timestamp("2024-10-01")
    filtered_df['Effective_Date'] = pd.to_datetime(
        np.where(
            pd.to_datetime(filtered_df['DELIVER_BY'], errors='coerce') >= cutoff_date,
            filtered_df['DELIVER_BY'],
            filtered_df['PICK_UP_BY']
        ),
        errors='coerce'
    )
    
    filtered_df['Month'] = filtered_df['Effective_Date'].dt.to_period('M')

    # Fetch coordinates only for the selected data
    def filter_and_enrich_city_coordinates(df, city_coords):
        # Combine unique origins and destinations into a single DataFrame
        relevant_origins = df[['ORIGCITY', 'ORIGPROV']].drop_duplicates()
        relevant_origins.rename(columns={"ORIGCITY": "CITY", "ORIGPROV": "PROVINCE"}, inplace=True)
        
        relevant_destinations = df[['DESTCITY', 'DESTPROV']].drop_duplicates()
        relevant_destinations.rename(columns={"DESTCITY": "CITY", "DESTPROV": "PROVINCE"}, inplace=True)
        
        relevant_cities = pd.concat([relevant_origins, relevant_destinations]).drop_duplicates()
        
        # Merge with city_coords to get coordinates for relevant cities
        enriched_cities = relevant_cities.merge(city_coords, on=["CITY", "PROVINCE"], how="inner")
        return enriched_cities
    
    # Filter city_coordinates_df for current month data
    enriched_coordinates = filter_and_enrich_city_coordinates(filtered_df, city_coordinates_df)

    # Calculate straight-line distance using haversine formula
    filtered_df['Straight Distance'] = calculate_haversine(filtered_df)

    punit_options = sorted(filtered_df['PICK_UP_PUNIT'].unique())
    selected_punit = st.selectbox("Select PUNIT:", options=punit_options)
    relevant_drivers = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit]['DRIVER_ID'].unique()
    driver_options = ["All"] + sorted(relevant_drivers.astype(str))
    selected_driver = st.selectbox("Select Driver ID (optional):", options=driver_options)
    filtered_view = filtered_df[filtered_df['PICK_UP_PUNIT'] == selected_punit].copy()
    if selected_driver != "All":
        filtered_view = filtered_view[filtered_view['DRIVER_ID'] == selected_driver].copy()

    months = sorted(filtered_view['Month'].unique())
    if len(months) == 0:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    else:
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()

    if not month_data.empty:
        # Ensure sorting by Effective_Date to maintain order
        month_data = month_data.sort_values(by='Effective_Date')
    
        # Assign a unique group for each day and alternate colors
        month_data['Day_Group'] = month_data['Effective_Date'].dt.date  # Extract only the date part
        unique_days = list(month_data['Day_Group'].unique())  # Get unique dates in the order they appear
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}  # Alternating colors: 0 or 1
        month_data['Highlight'] = month_data['Day_Group'].map(day_colors)  # Map each day to its alternating color
    
        # Identify missing coordinates for origin and destination
        missing_origins = month_data[pd.isna(month_data['ORIG_LAT'])][['ORIGCITY', 'ORIGPROV']].drop_duplicates()
        missing_destinations = month_data[pd.isna(month_data['DEST_LAT'])][['DESTCITY', 'DESTPROV']].drop_duplicates()
    
        # Combine missing cities
        missing_cities = pd.concat([
            missing_origins.rename(columns={'ORIGCITY': 'City', 'ORIGPROV': 'Province'}),
            missing_destinations.rename(columns={'DESTCITY': 'City', 'DESTPROV': 'Province'})
        ]).drop_duplicates()
    
        # Add missing cities to the table
        month_data['Straight Distance'] = np.where(
            pd.isna(month_data['ORIG_LAT']) | pd.isna(month_data['DEST_LAT']),
            np.nan,
            month_data['Straight Distance']
        )
    
        # Create the route summary DataFrame
        month_data['Profit (CAD)'] = month_data['TOTAL_CHARGE_CAD'] - month_data['TOTAL_PAY_AMT']
    
        route_summary_df = month_data.assign(
            Route=lambda x: x['ORIGCITY'] + ", " + x['ORIGPROV'] + " to " + x['DESTCITY'] + ", " + x['DESTPROV']
        )[[  # Include "Highlight" for styling
            "Route", "BILL_NUMBER", "TOTAL_CHARGE_CAD", "DISTANCE", "Straight Distance", 
            "Revenue per Mile", "DRIVER_ID", "TOTAL_PAY_AMT", "Profit (CAD)", "Effective_Date", "Highlight"
        ]].rename(columns={
            "TOTAL_CHARGE_CAD": "Total Charge (CAD)", 
            "DISTANCE": "Distance (miles)", 
            "Straight Distance": "Straight Distance (miles)", 
            "TOTAL_PAY_AMT": "Driver Pay (CAD)"
        })
    
        # Calculate grand totals
        grand_totals = pd.DataFrame([{
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Total Charge (CAD)": route_summary_df["Total Charge (CAD)"].sum(),
            "Distance (miles)": route_summary_df["Distance (miles)"].sum(),
            "Straight Distance (miles)": route_summary_df["Straight Distance (miles)"].sum(),
            "Revenue per Mile": route_summary_df["Total Charge (CAD)"].sum() / route_summary_df["Distance (miles)"].sum()
            if route_summary_df["Distance (miles)"].sum() != 0 else 0,
            "Driver Pay (CAD)": route_summary_df["Driver Pay (CAD)"].sum(),
            "Profit (CAD)": route_summary_df["Total Charge (CAD)"].sum() - route_summary_df["Driver Pay (CAD)"].sum(),
            "Effective_Date": "",
            "Highlight": None  # No highlight for the grand totals row
        }])
    
        # Append grand totals to the route summary table
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
    
        # Format currency columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
            route_summary_df[col] = route_summary_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )
    
        # Define row styling
        def highlight_rows(row):
            if row['Route'] == "Grand Totals":
                return ['background-color: #f7c8c8'] * len(row)  # Highlight grand totals in red
            elif row['Highlight'] == 1:
                return ['background-color: #c8e0f7'] * len(row)  # One color for odd groups
            else:
                return ['background-color: #f7f7c8'] * len(row)  # Another color for even groups
    
        # Apply styling and display the DataFrame
        styled_route_summary = route_summary_df.style.apply(highlight_rows, axis=1)
        st.write("Route Summary:")
        st.dataframe(styled_route_summary, use_container_width=True)
    
        # Generate the map
        fig = go.Figure()
    
        # Sequential numbering logic for origin and destination
        city_sequence = {city: [] for city in set(month_data['ORIGCITY']).union(month_data['DESTCITY'])}
        label_counter = 1
        for _, row in month_data.iterrows():
            city_sequence[row['ORIGCITY']].append(label_counter)
            label_counter += 1
            city_sequence[row['DESTCITY']].append(label_counter)
            label_counter += 1
    
        # Track if legend entries have been added
        legend_added = {"Origin": False, "Destination": False, "Route": False}
    
        for _, row in month_data.iterrows():
            if pd.notna(row['ORIG_LAT']) and pd.notna(row['DEST_LAT']):
                origin_sequence = ", ".join(map(str, city_sequence[row['ORIGCITY']]))
                destination_sequence = ", ".join(map(str, city_sequence[row['DESTCITY']]))
    
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
                    hovertext=(f"City: {row['ORIGCITY']}, {row['ORIGPROV']}<br>"
                               f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                               f"Distance (miles): {row['DISTANCE']:.1f}<br>"
                               f"Revenue per Mile: ${row['Revenue per Mile']:.2f}<br>"
                               f"Driver Pay (CAD): ${row['TOTAL_PAY_AMT']:.2f}<br>"
                               f"Profit (CAD): ${row['Profit (CAD)']:.2f}"),
                    showlegend=not legend_added["Origin"]
                ))
                legend_added["Origin"] = True
    
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
                    hovertext=(f"City: {row['DESTCITY']}, {row['DESTPROV']}<br>"
                               f"Total Charge (CAD): ${row['TOTAL_CHARGE_CAD']:.2f}<br>"
                               f"Distance (miles): {row['DISTANCE']:.1f}<br>"
                               f"Revenue per Mile: ${row['Revenue per Mile']:.2f}<br>"
                               f"Driver Pay (CAD): ${row['TOTAL_PAY_AMT']:.2f}<br>"
                               f"Profit (CAD): ${row['Profit (CAD)']:.2f}"),
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
    
        # Update map layout
        fig.update_layout(
            title=f"Routes for {selected_month} - PUNIT: {selected_punit}, Driver ID: {selected_driver}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)
    
        # Display missing cities
        if not missing_cities.empty:
            st.subheader("Cities missing coordinates")
            st.write(missing_cities)
    else:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    
else:
    st.warning("Please upload both the TLORDER and DRIVERPAY CSV files to proceed.")
