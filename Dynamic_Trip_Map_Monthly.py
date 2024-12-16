import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re

# Initialize global variables for navigation
if "month_index" not in st.session_state:
    st.session_state.month_index = 0

# Streamlit App Title and Instructions
st.title("Trip Map Viewer by Month")

st.markdown("""
### Instructions:
Use the following query to generate the required TLORDER data:  
SELECT BILL_NUMBER, DETAIL_LINE_ID, CALLNAME, ORIGCITY, ORIGPROV, DESTCITY, DESTPROV, PICK_UP_PUNIT, DELIVERY_PUNIT, CHARGES, XCHARGES, DISTANCE, DISTANCE_UNITS, CURRENCY_CODE, PICK_UP_BY, DELIVER_BY  
FROM TLORDER WHERE "PICK_UP_BY" BETWEEN 'X' AND 'Y';  

Use the following query to generate the required DRIVERPAY data:  
SELECT BILL_NUMBER, PAY_ID, DRIVER_ID, PAY_DESCRIPTION, FB_TOTAL_CHARGES, CURRENCY_CODE, TOTAL_PAY_AMT, PAID_DATE, DATE_TRANS  
FROM DRIVERPAY WHERE "PAID_DATE" BETWEEN 'X' AND 'Y';  

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
    
    # Define a function to clean city names (keep letters, spaces only)
    def clean_city_name(name):
        return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().upper()

    # Clean and standardize city and province names to uppercase in both datasets
    city_coords['CITY'] = city_coords['CITY'].apply(clean_city_name)
    city_coords['PROVINCE'] = city_coords['PROVINCE'].str.strip().str.upper()
    
    df['ORIGCITY'] = df['ORIGCITY'].apply(clean_city_name)
    df['ORIGPROV'] = df['ORIGPROV'].str.strip().str.upper()
    df['DESTCITY'] = df['DESTCITY'].apply(clean_city_name)
    df['DESTPROV'] = df['DESTPROV'].str.strip().str.upper()

    # Ensure there are no duplicates in city_coords after cleaning
    city_coords = city_coords.drop_duplicates(subset=['CITY', 'PROVINCE'])

    # Merge for origins
    origin_coords = city_coords.rename(columns={"CITY": "ORIGCITY", "PROVINCE": "ORIGPROV", "LAT": "ORIG_LAT", "LON": "ORIG_LON"})
    df = df.merge(origin_coords, on=["ORIGCITY", "ORIGPROV"], how="left")
    
    # Merge for destinations
    dest_coords = city_coords.rename(columns={"CITY": "DESTCITY", "PROVINCE": "DESTPROV", "LAT": "DEST_LAT", "LON": "DEST_LON"})
    df = df.merge(dest_coords, on=["DESTCITY", "DESTPROV"], how="left")
    
    return df

@st.cache_data
def filter_and_enrich_city_coordinates(df, city_coords):
    # Define a function to clean city names (keep letters, spaces only)
    def clean_city_name(name):
        return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().upper()

    # Clean and standardize city and province names to uppercase
    city_coords['CITY'] = city_coords['CITY'].apply(clean_city_name)
    city_coords['PROVINCE'] = city_coords['PROVINCE'].str.strip().str.upper()

    # Combine unique origins and destinations into a single DataFrame
    relevant_origins = df[['ORIGCITY', 'ORIGPROV']].drop_duplicates()
    relevant_origins.rename(columns={"ORIGCITY": "CITY", "ORIGPROV": "PROVINCE"}, inplace=True)

    relevant_destinations = df[['DESTCITY', 'DESTPROV']].drop_duplicates()
    relevant_destinations.rename(columns={"DESTCITY": "CITY", "DESTPROV": "PROVINCE"}, inplace=True)

    relevant_cities = pd.concat([relevant_origins, relevant_destinations]).drop_duplicates()

    # Merge with city_coords to get coordinates for relevant cities
    enriched_cities = relevant_cities.merge(city_coords, on=["CITY", "PROVINCE"], how="left")

    # Remove duplicates from the final enriched cities
    enriched_cities = enriched_cities.drop_duplicates()

    return enriched_cities

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

    # Merge TLORDER and DRIVERPAY data
    tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

    # Filter for valid routes, but do not exclude rows missing coordinates
    valid_routes = tlorder_df[
        (tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) &
        (pd.notna(tlorder_df['DISTANCE']))
    ].copy()

    exchange_rate = 1.38
    valid_routes['CHARGES'] = pd.to_numeric(valid_routes['CHARGES'], errors='coerce')
    valid_routes['XCHARGES'] = pd.to_numeric(valid_routes['XCHARGES'], errors='coerce')
    valid_routes['TOTAL_CHARGE_CAD'] = np.where(
        valid_routes['CURRENCY_CODE'] == 'USD',
        (valid_routes['CHARGES'] + valid_routes['XCHARGES']) * exchange_rate,
        valid_routes['CHARGES'] + valid_routes['XCHARGES']
    )

    # Filter for non-zero charges
    filtered_df = valid_routes[
        (valid_routes['TOTAL_CHARGE_CAD'] != 0)
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

    # Find cities missing in the coordinates dataset
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

    missing_cities = pd.concat([missing_origins, missing_destinations]).drop_duplicates()

    # Fill missing Straight Distance as np.nan for rows missing coordinates
    filtered_df['Straight Distance'] = np.where(
        pd.isna(filtered_df['ORIG_LAT']) | pd.isna(filtered_df['DEST_LAT']),
        np.nan,
        calculate_haversine(filtered_df)
    )

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
        # Assign colors for alternating rows by day
        month_data = month_data.sort_values(by='Effective_Date')
        month_data['Day_Group'] = month_data['Effective_Date'].dt.date
        unique_days = list(month_data['Day_Group'].unique())
        day_colors = {day: idx % 2 for idx, day in enumerate(unique_days)}
        month_data['Highlight'] = month_data['Day_Group'].map(day_colors)
    
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
            "Highlight": None
        }])
    
        route_summary_df = pd.concat([route_summary_df, grand_totals], ignore_index=True)
    
        # Format currency columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
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
    
        # Generate the map
        fig = go.Figure()

        # Aggregate totals for each city and province combination
        aggregated_origins = month_data.groupby(['ORIGCITY', 'ORIGPROV']).agg({
            'Total Charge (CAD)': 'sum',
            'Distance (miles)': 'sum',
            'Driver Pay (CAD)': 'sum',
            'Profit (CAD)': 'sum'
        }).reset_index()
        
        aggregated_destinations = month_data.groupby(['DESTCITY', 'DESTPROV']).agg({
            'Total Charge (CAD)': 'sum',
            'Distance (miles)': 'sum',
            'Driver Pay (CAD)': 'sum',
            'Profit (CAD)': 'sum'
        }).reset_index()
        
        # Combine origins and destinations, keeping totals by city
        aggregated_origins.rename(columns={'ORIGCITY': 'City', 'ORIGPROV': 'Province'}, inplace=True)
        aggregated_destinations.rename(columns={'DESTCITY': 'City', 'DESTPROV': 'Province'}, inplace=True)
        aggregated_totals = pd.concat([aggregated_origins, aggregated_destinations]).groupby(['City', 'Province']).sum().reset_index()
        
        # Sequential numbering logic for cities
        city_sequence = {city: [] for city in aggregated_totals['City']}
        label_counter = 1
        for city in city_sequence.keys():
            city_sequence[city].append(label_counter)
            label_counter += 1
        
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        
        for _, row in month_data.iterrows():
            origin_sequence = ", ".join(map(str, city_sequence[row['ORIGCITY']]))
            destination_sequence = ", ".join(map(str, city_sequence[row['DESTCITY']]))

            # Add origin marker
            for _, row in aggregated_totals.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[None],  # Placeholder as no coordinates are being plotted here
                lat=[None],
                mode="markers+text",
                marker=dict(size=8, color="blue"),  # Marker color
                text=f"{city_sequence[row['City']][0]}",
                textposition="top right",
                name="City Totals",
                hoverinfo="text",
                hovertext=(f"City: {row['City']}, {row['Province']}<br>"
                           f"Total Charge (CAD): ${row['Total Charge (CAD)']:.2f}<br>"
                           f"Distance (miles): {row['Distance (miles)']:.1f}<br>"
                           f"Revenue per Mile: ${(row['Total Charge (CAD)'] / row['Distance (miles)']):.2f}<br>"
                           f"Driver Pay (CAD): ${row['Driver Pay (CAD)']:.2f}<br>"
                           f"Profit (CAD): ${row['Profit (CAD)']:.2f}")
            ))
            legend_added["Origin"] = True

            # Add destination marker
            for _, row in aggregated_totals.iterrows():
            fig.add_trace(go.Scattergeo(
                lon=[None],  # Placeholder as no coordinates are being plotted here
                lat=[None],
                mode="markers+text",
                marker=dict(size=8, color="blue"),  # Marker color
                text=f"{city_sequence[row['City']][0]}",
                textposition="top right",
                name="City Totals",
                hoverinfo="text",
                hovertext=(f"City: {row['City']}, {row['Province']}<br>"
                           f"Total Charge (CAD): ${row['Total Charge (CAD)']:.2f}<br>"
                           f"Distance (miles): {row['Distance (miles)']:.1f}<br>"
                           f"Revenue per Mile: ${(row['Total Charge (CAD)'] / row['Distance (miles)']):.2f}<br>"
                           f"Driver Pay (CAD): ${row['Driver Pay (CAD)']:.2f}<br>"
                           f"Profit (CAD): ${row['Profit (CAD)']:.2f}")
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
            title=f"Routes for {selected_month} - PUNIT: {selected_punit}, Driver ID: {selected_driver}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)

        # Display cities missing coordinates relevant to the selection
        relevant_missing_origins = month_data[
            (pd.isna(month_data['ORIG_LAT']) | pd.isna(month_data['ORIG_LON']))
        ][['ORIGCITY', 'ORIGPROV']].drop_duplicates().rename(columns={
            'ORIGCITY': 'City', 'ORIGPROV': 'Province'
        })

        relevant_missing_destinations = month_data[
            (pd.isna(month_data['DEST_LAT']) | pd.isna(month_data['DEST_LON']))
        ][['DESTCITY', 'DESTPROV']].drop_duplicates().rename(columns={
            'DESTCITY': 'City', 'DESTPROV': 'Province'
        })

        relevant_missing_cities = pd.concat([relevant_missing_origins, relevant_missing_destinations]).drop_duplicates()

        if not relevant_missing_cities.empty:
            st.write("### Cities Missing Coordinates")
            st.dataframe(relevant_missing_cities, use_container_width=True)
    else:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    
else:
    st.warning("Please upload both the TLORDER and DRIVERPAY CSV files to proceed.")
    
