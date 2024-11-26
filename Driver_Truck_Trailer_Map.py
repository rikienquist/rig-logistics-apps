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
    # Load uploaded data
    try:
        df = pd.read_excel(uploaded_file, sheet_name="attachment")
    except Exception as e:
        st.error(f"Failed to load the uploaded file: {e}")
        st.stop()

    # Load coordinates data
    coordinates_file = "trip_map_data/legsum_coordinates.csv"
    try:
        coordinates_df = pd.read_csv(coordinates_file)
    except Exception as e:
        st.error(f"Failed to load coordinates data: {e}")
        st.stop()

    # Merge coordinates with the main data
    df = df.merge(
        coordinates_df.rename(columns={"Location": "LEGO_ZONE_DESC", "Latitude": "LEGO_LAT", "Longitude": "LEGO_LON"}),
        on="LEGO_ZONE_DESC",
        how="left"
    )
    df = df.merge(
        coordinates_df.rename(columns={"Location": "LEGD_ZONE_DESC", "Latitude": "LEGD_LAT", "Longitude": "LEGD_LON"}),
        on="LEGD_ZONE_DESC",
        how="left"
    )

    # Sidebar Filters with Dynamic Options
    st.sidebar.header("Filters")

    # Truck Unit filter
    truck_options = sorted(df['LS_POWER_UNIT'].dropna().unique())
    selected_truck = st.sidebar.selectbox("Select Truck Unit:", ["None"] + list(truck_options))

    # Update Driver ID options based on Truck Unit selection
    if selected_truck != "None":
        relevant_drivers = df[df['LS_POWER_UNIT'] == selected_truck]['LS_DRIVER'].dropna().unique()
        relevant_trailers = df[df['LS_POWER_UNIT'] == selected_truck]['LS_TRAILER1'].dropna().unique()
    else:
        relevant_drivers = []
        relevant_trailers = []

    # Driver ID filter
    selected_driver = st.sidebar.selectbox(
        "Select Driver ID:", 
        ["None"] + sorted(relevant_drivers) if selected_truck != "None" else ["None"]
    )

    # Update Trailer Unit options based on Truck Unit and Driver ID selection
    if selected_driver != "None" and selected_truck != "None":
        relevant_trailers = df[(df['LS_POWER_UNIT'] == selected_truck) & (df['LS_DRIVER'] == selected_driver)]['LS_TRAILER1'].dropna().unique()

    # Trailer Unit filter
    selected_trailer = st.sidebar.selectbox(
        "Select Trailer Unit:", 
        ["None"] + sorted(relevant_trailers) if selected_truck != "None" else ["None"]
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_truck != "None":
        filtered_df = filtered_df[filtered_df["LS_POWER_UNIT"] == selected_truck]
    if selected_driver != "None":
        filtered_df = filtered_df[filtered_df["LS_DRIVER"] == selected_driver]
    if selected_trailer != "None":
        filtered_df = filtered_df[filtered_df["LS_TRAILER1"] == selected_trailer]

    # Map and Table Visualization
    if not filtered_df.empty:
        # Sort by INS_TIMESTAMP
        filtered_df = filtered_df.sort_values(by="INS_TIMESTAMP").reset_index(drop=True)

        # Map Visualization
        fig = go.Figure()
        legend_added = {"Origin": False, "Destination": False, "Route": False}
        sequence_counter = 1

        for _, row in filtered_df.iterrows():
            # Add origin marker
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON']],
                lat=[row['LEGO_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="blue"),
                text=str(sequence_counter),
                name="Origin" if not legend_added["Origin"] else None,
                hoverinfo="text",
                hovertext=row['LEGO_ZONE_DESC'],
                showlegend=not legend_added["Origin"]
            ))
            legend_added["Origin"] = True
            sequence_counter += 1

            # Add destination marker
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGD_LON']],
                lat=[row['LEGD_LAT']],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=str(sequence_counter),
                name="Destination" if not legend_added["Destination"] else None,
                hoverinfo="text",
                hovertext=row['LEGD_ZONE_DESC'],
                showlegend=not legend_added["Destination"]
            ))
            legend_added["Destination"] = True
            sequence_counter += 1

            # Add route line
            fig.add_trace(go.Scattergeo(
                lon=[row['LEGO_LON'], row['LEGD_LON']],
                lat=[row['LEGO_LAT'], row['LEGD_LAT']],
                mode="lines",
                line=dict(width=2, color="green"),
                name="Route" if not legend_added["Route"] else None,
                hoverinfo="none",
                showlegend=not legend_added["Route"]
            ))
            legend_added["Route"] = True

        fig.update_layout(
            title="Driver, Truck, and Trailer Movement Map",
            geo=dict(scope="north america", projection_type="mercator"),
            showlegend=True
        )

        st.plotly_chart(fig)

        # Details Table
        st.write("Details:")
        filtered_df['INS_TIMESTAMP'] = pd.to_datetime(filtered_df['INS_TIMESTAMP'])
        st.dataframe(filtered_df[[
            "LS_DRIVER", "LS_POWER_UNIT", "LS_TRAILER1", "LEGO_ZONE_DESC", "LEGD_ZONE_DESC", 
            "LS_TO_ZONE", "LS_LEG_DIST", "LS_MT_LOADED", "INS_TIMESTAMP", "LS_LEG_NOTE"
        ]])
    else:
        st.warning("No data available for the selected filters.")
else:
    st.info("Please upload a file to proceed.")
