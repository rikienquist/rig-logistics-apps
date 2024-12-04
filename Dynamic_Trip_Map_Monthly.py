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

# Optimized function to load city coordinates with caching
@st.cache_data
def load_city_coordinates():
    city_coords = pd.read_csv("trip_map_data/city_coordinates.csv")
    city_coords = city_coords.rename(columns={
        "city": "CITY",
        "province": "PROVINCE",
        "latitude": "LAT",
        "longitude": "LON"
    })
    return city_coords

# Optimized function to preprocess TLORDER data
@st.cache_data
def preprocess_tlorder(file, city_coords):
    df = pd.read_csv(file, low_memory=False)
    # Merge for origins
    df = df.merge(city_coords.rename(columns={"CITY": "ORIGCITY", "PROVINCE": "ORIGPROV", "LAT": "ORIG_LAT", "LON": "ORIG_LON"}), 
                  on=["ORIGCITY", "ORIGPROV"], how="left")
    # Merge for destinations
    df = df.merge(city_coords.rename(columns={"CITY": "DESTCITY", "PROVINCE": "DESTPROV", "LAT": "DEST_LAT", "LON": "DEST_LON"}), 
                  on=["DESTCITY", "DESTPROV"], how="left")
    return df

# Optimized function to preprocess DRIVERPAY data
@st.cache_data
def preprocess_driverpay(file):
    df = pd.read_csv(file, low_memory=False)
    df['TOTAL_PAY_AMT'] = pd.to_numeric(df['TOTAL_PAY_AMT'], errors='coerce')
    driver_pay_agg = df.groupby('BILL_NUMBER').agg({
        'TOTAL_PAY_AMT': 'sum',
        'DRIVER_ID': 'first'
    }).reset_index()
    return driver_pay_agg

if uploaded_tlorder_file and uploaded_driverpay_file:
    # Load city coordinates
    city_coordinates_df = load_city_coordinates()
    
    # Preprocess TLORDER and DRIVERPAY data
    tlorder_df = preprocess_tlorder(uploaded_tlorder_file, city_coordinates_df)
    driver_pay_agg = preprocess_driverpay(uploaded_driverpay_file)

    # Merge the preprocessed TLORDER data with aggregated DRIVERPAY data
    tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

    # Filter for valid routes
    tlorder_df = tlorder_df[(tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']) &
                            (pd.notna(tlorder_df['ORIG_LAT'])) & 
                            (pd.notna(tlorder_df['DEST_LAT']))].copy()

    # Calculate total charge in CAD
    tlorder_df['CHARGES'] = pd.to_numeric(tlorder_df['CHARGES'], errors='coerce')
    tlorder_df['XCHARGES'] = pd.to_numeric(tlorder_df['XCHARGES'], errors='coerce')
    tlorder_df['TOTAL_CHARGE_CAD'] = np.where(
        tlorder_df['CURRENCY_CODE'] == 'USD',
        (tlorder_df['CHARGES'] + tlorder_df['XCHARGES']) * 1.38,
        tlorder_df['CHARGES'] + tlorder_df['XCHARGES']
    )

    # Filter for non-zero charges and distance
    filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & 
                             (tlorder_df['DISTANCE'] != 0)].copy()

    # Ensure `PICK_UP_PUNIT` is clean
    filtered_df['PICK_UP_PUNIT'] = filtered_df['PICK_UP_PUNIT'].astype(str).fillna("Unknown")

    # Calculate Revenue per Mile and Profit
    filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
    filtered_df['Profit (CAD)'] = filtered_df['TOTAL_CHARGE_CAD'] - filtered_df['TOTAL_PAY_AMT']

    # Add a new column for dynamic date selection
    cutoff_date = pd.Timestamp("2024-10-01")
    filtered_df['Effective_Date'] = pd.to_datetime(
        np.where(
            pd.to_datetime(filtered_df['DELIVER_BY'], errors='coerce') >= cutoff_date,
            filtered_df['DELIVER_BY'],
            filtered_df['PICK_UP_BY']
        ),
        errors='coerce'
    )
    
    # Ensure the effective date is in datetime format
    filtered_df['Month'] = filtered_df['Effective_Date'].dt.to_period('M')

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
    
    # Month dropdown for selection
    months = sorted(filtered_view['Month'].dropna().unique())
    
    # If no months are available, handle gracefully
    if len(months) == 0:
        st.warning("No data available for the selected PUNIT and Driver ID.")
    else:
        # Ensure the selected month is valid for the current data
        if "selected_month" not in st.session_state or st.session_state.selected_month not in months:
            st.session_state.selected_month = months[0]  # Default to the first month if invalid or not set
    
        # Render the dropdown with dynamic options
        selected_month = st.selectbox("Select Month:", options=months, index=months.index(st.session_state.selected_month))
        st.session_state.selected_month = selected_month
    
        # Filter data for the selected month
        month_data = filtered_view[filtered_view['Month'] == selected_month].copy()

    if not month_data.empty:
        # Create the route summary and map as before
        month_data['Highlight'] = (month_data['Effective_Date'].dt.date != month_data['Effective_Date'].dt.date.shift()).cumsum() % 2

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
                "Profit (CAD)": row['TOTAL_CHARGE_CAD'] - row['TOTAL_PAY_AMT'],
                "Date": row['Effective_Date']
            })

        # Convert the route summary to a DataFrame
        route_summary_df = pd.DataFrame(route_summary)

        # Calculate grand totals
        total_charge = route_summary_df["Total Charge (CAD)"].sum()
        total_distance = route_summary_df["Distance (miles)"].sum()
        total_straight_distance = route_summary_df["Straight Distance (miles)"].sum()
        total_driver_pay = route_summary_df["Driver Pay (CAD)"].sum()
        total_profit = total_charge - total_driver_pay
        grand_revenue_per_mile = total_charge / total_distance if total_distance != 0 else 0

        # Add grand totals row
        grand_totals = {
            "Route": "Grand Totals",
            "BILL_NUMBER": "",
            "Total Charge (CAD)": total_charge,
            "Distance (miles)": total_distance,
            "Straight Distance (miles)": total_straight_distance,
            "Revenue per Mile": grand_revenue_per_mile,
            "Driver ID": "",
            "Driver Pay (CAD)": total_driver_pay,
            "Profit (CAD)": total_profit,
            "Date": ""
        }

        # Append grand totals to DataFrame
        route_summary_df = pd.concat([route_summary_df, pd.DataFrame([grand_totals])], ignore_index=True)

        # Format currency columns
        for col in ["Total Charge (CAD)", "Revenue per Mile", "Driver Pay (CAD)", "Profit (CAD)"]:
            route_summary_df[col] = route_summary_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and isinstance(x, (float, int)) else x
            )

        # Highlight rows for alternate days
        def highlight_rows(row):
            if row['Route'] == "Grand Totals":
                return ['background-color: #f7c8c8'] * len(row)
            elif row['Date'] != "":
                date_highlight = month_data.loc[month_data['Effective_Date'] == row['Date'], 'Highlight'].iloc[0]
                return ['background-color: #c8e0f7' if date_highlight == 1 else 'background-color: #f7f7c8'] * len(row)
            else:
                return ['background-color: white'] * len(row)

        # Apply styling and display the DataFrame
        styled_route_summary = route_summary_df.style.apply(highlight_rows, axis=1)
        st.write("Route Summary:")
        st.dataframe(styled_route_summary, use_container_width=True)

        # Generate the map
        fig = go.Figure()
        month_data = month_data.sort_values(by='Effective_Date')  # Sort routes chronologically
        label_counter = 1
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        for _, row in month_data.iterrows():
            # Add origin marker
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
                           f"Date: {row['Effective_Date']}<br>"
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
                marker=dict(size=10, color="red"),
                text=str(label_counter + 1),
                textposition="top right",
                name="Destination" if not legend_added["Destination"] else None,
                hoverinfo="text",
                hovertext=(f"City: {row['DESTCITY']}, {row['DESTPROV']}<br>"
                           f"Date: {row['Effective_Date']}<br>"
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
        
            label_counter += 2
        
        # Update map layout
        fig.update_layout(
            title=f"Routes for {selected_month} - PUNIT: {selected_punit}, Driver ID: {selected_driver or 'All'}",
            geo=dict(scope="north america", projection_type="mercator"),
        )
        st.plotly_chart(fig)

    else:
        st.warning("No data available for the selected PUNIT and Driver ID.")

else:
    st.warning("Please upload both the TLORDER and DRIVERPAY CSV files to proceed.")
