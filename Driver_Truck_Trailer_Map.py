import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Set up the Streamlit app
st.title("Driver, Truck and Trailer Movement Map")

# Instructions for the user
st.markdown("""
**Instructions**:
1. Use the following query to generate the required LEGSUM data:

SELECT LS_DRIVER, LS_POWER_UNIT, LS_TRAILER1, LEGO_ZONE_DESC, LEGD_ZONE_DESC, LS_TO_ZONE, LS_LEG_DIST, LS_MT_LOADED, LS_TO_ZONE, LS_FREIGHT, INS_TIMESTAMP, LS_LEG_NOTE FROM LEGSUM WHERE "INS_TIMESTAMP" BETWEEN 'X' AND 'Y';

Replace `X` and `Y` with the desired date range.

2. Save the query results as an Excel file with the sheet name `attachment`.

3. Upload the file below to visualize the data.

""")

# File upload
uploaded_file = st.file_uploader("Upload the LEGSUM Excel file:", type=["xlsx"])
if uploaded_file:
    # Read uploaded Excel file
    try:
        df = pd.read_excel(uploaded_file, sheet_name="attachment")
    except Exception as e:
        st.error(f"Failed to read the uploaded file: {e}")
        st.stop()
    
    # Load coordinates data
    coordinates_file = "trip_map_data/legsum_coordinates.csv"
    try:
        coordinates_df = pd.read_csv(coordinates_file)
    except Exception as e:
        st.error(f"Failed to load coordinates data: {e}")
        st.stop()
    
    # Merge coordinates with the main data
    df = df.merge(coordinates_df.rename(columns={"Location": "LEGO_ZONE_DESC", "Latitude": "LEGO_LAT", "Longitude": "LEGO_LON"}), 
                  on="LEGO_ZONE_DESC", how="left")
    df = df.merge(coordinates_df.rename(columns={"Location": "LEGD_ZONE_DESC", "Latitude": "LEGD_LAT", "Longitude": "LEGD_LON"}), 
                  on="LEGD_ZONE_DESC", how="left")
    
    # Check for missing coordinates
    missing_coords = df[(df['LEGO_LAT'].isna()) | (df['LEGD_LAT'].isna())]
    if not missing_coords.empty:
        st.warning("Some locations are missing coordinates. Please update the coordinates file.")
        st.dataframe(missing_coords[["LEGO_ZONE_DESC", "LEGD_ZONE_DESC"]])
    
    # Filter options
    drivers = df['LS_DRIVER'].dropna().unique()
    trucks = df['LS_POWER_UNIT'].dropna().unique()
    trailers = df['LS_TRAILER1'].dropna().unique()
    
    st.sidebar.header("Filters")
    selected_driver = st.sidebar.selectbox("Select Driver ID:", ["All"] + list(drivers))
    selected_truck = st.sidebar.selectbox("Select Truck Unit:", ["All"] + list(trucks))
    selected_trailer = st.sidebar.selectbox("Select Trailer Unit:", ["All"] + list(trailers))
    
    # Filter the DataFrame
    filtered_df = df.copy()
    if selected_driver != "All":
        filtered_df = filtered_df[filtered_df["LS_DRIVER"] == selected_driver]
    if selected_truck != "All":
        filtered_df = filtered_df[filtered_df["LS_POWER_UNIT"] == selected_truck]
    if selected_trailer != "All":
        filtered_df = filtered_df[filtered_df["LS_TRAILER1"] == selected_trailer]
    
    # Display the map
    if not filtered_df.empty:
        fig = go.Figure()
        for _, row in filtered_df.iterrows():
            # Add markers for origin
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON']],
                lat=[row['LEGO_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="blue"),
                text=row['LEGO_ZONE_DESC'],
                name="Origin",
                hoverinfo="text"
            ))
            # Add markers for destination
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGD_LON']],
                lat=[row['LEGD_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=row['LEGD_ZONE_DESC'],
                name="Destination",
                hoverinfo="text"
            ))
            # Add route line
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON'], row['LEGD_LON']],
                lat=[row['LEGO_LAT'], row['LEGD_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route",
                hoverinfo="none"
            ))
        
        fig.update_layout(
            title="Movement Map",
            geo=dict(scope="north america", projection_type="mercator"),
            showlegend=True
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected filters.")
    
    # Display data table
    st.write("Movement Details Table:")
    st.dataframe(filtered_df[[
        "LS_DRIVER", "LS_POWER_UNIT", "LS_TRAILER1", "LEGO_ZONE_DESC", "LEGD_ZONE_DESC", 
        "LS_TO_ZONE", "LS_LEG_DIST", "LS_MT_LOADED", "INS_TIMESTAMP", "LS_LEG_NOTE"
    ]])
else:
    st.info("Please upload a file to proceed.")
