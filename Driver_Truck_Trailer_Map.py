import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Set up the Streamlit app
st.title("Driver, Truck and Trailer Movement Map")

# Instructions for the user
st.markdown("""
**Instructions**:
1. Use the following query to generate the required LEGSUM data:
Replace `X` and `Y` with the desired date range.
2. Save the query results as an Excel file with the sheet name `attachment`.
3. Upload the file below to visualize the data.
""")

# File uploader
uploaded_file = st.file_uploader("Upload the LEGSUM Excel file", type=["xlsx"])
if uploaded_file is not None:
 # Load the data
 try:
     data = pd.read_excel(uploaded_file, sheet_name="attachment")
     coordinates = pd.read_csv("legsum_coordinates.csv")  # Load coordinates file
     
     # Merge the coordinates with LEGO_ZONE_DESC and LEGD_ZONE_DESC
     data = data.merge(coordinates, left_on="LEGO_ZONE_DESC", right_on="Location", how="left").rename(
         columns={"Latitude": "LEGO_LAT", "Longitude": "LEGO_LON"}
     )
     data = data.merge(coordinates, left_on="LEGD_ZONE_DESC", right_on="Location", how="left").rename(
         columns={"Latitude": "LEGD_LAT", "Longitude": "LEGD_LON"}
     )
     data.drop(columns=["Location"], inplace=True)

     # Check for missing coordinates
     missing_coords = data[data["LEGO_LAT"].isna() | data["LEGD_LAT"].isna()]
     if not missing_coords.empty:
         st.warning("Some locations are missing coordinates. Please update 'legsum_coordinates.csv' and re-upload.")
         st.dataframe(missing_coords)
     else:
         # Filters
         st.sidebar.header("Filters")
         drivers = ["All"] + sorted(data["LS_DRIVER"].dropna().unique())
         selected_driver = st.sidebar.selectbox("Select Driver ID (LS_DRIVER):", drivers)

         trucks = ["All"] + sorted(data["LS_POWER_UNIT"].dropna().unique())
         selected_truck = st.sidebar.selectbox("Select Truck Unit (LS_POWER_UNIT):", trucks)

         trailers = ["All"] + sorted(data["LS_TRAILER1"].dropna().unique())
         selected_trailer = st.sidebar.selectbox("Select Trailer Unit (LS_TRAILER1):", trailers)

         # Apply filters
         filtered_data = data.copy()
         if selected_driver != "All":
             filtered_data = filtered_data[filtered_data["LS_DRIVER"] == selected_driver]
         if selected_truck != "All":
             filtered_data = filtered_data[filtered_data["LS_POWER_UNIT"] == selected_truck]
         if selected_trailer != "All":
             filtered_data = filtered_data[filtered_data["LS_TRAILER1"] == selected_trailer]

         if filtered_data.empty:
             st.warning("No data available for the selected filters.")
         else:
             # Map visualization
             st.subheader("Movement Map")
             fig = go.Figure()

             for _, row in filtered_data.iterrows():
                 # Add route
                 fig.add_trace(go.Scattergeo(
                     lon=[row["LEGO_LON"], row["LEGD_LON"]],
                     lat=[row["LEGO_LAT"], row["LEGD_LAT"]],
                     mode="lines",
                     line=dict(width=2, color="blue"),
                     name=f"Route: {row['LEGO_ZONE_DESC']} → {row['LEGD_ZONE_DESC']}",
                     hoverinfo="text",
                     hovertext=(f"Driver: {row['LS_DRIVER']}<br>"
                                f"Truck: {row['LS_POWER_UNIT']}<br>"
                                f"Trailer: {row['LS_TRAILER1']}<br>"
                                f"Distance: {row['LS_LEG_DIST']} miles")
                 ))
                 # Add origin marker
                 fig.add_trace(go.Scattergeo(
                     lon=[row["LEGO_LON"]],
                     lat=[row["LEGO_LAT"]],
                     mode="markers",
                     marker=dict(size=8, color="green"),
                     name="Origin",
                     hoverinfo="text",
                     hovertext=f"Origin: {row['LEGO_ZONE_DESC']}"
                 ))
                 # Add destination marker
                 fig.add_trace(go.Scattergeo(
                     lon=[row["LEGD_LON"]],
                     lat=[row["LEGD_LAT"]],
                     mode="markers",
                     marker=dict(size=8, color="red"),
                     name="Destination",
                     hoverinfo="text",
                     hovertext=f"Destination: {row['LEGD_ZONE_DESC']}"
                 ))

             fig.update_layout(
                 title="Driver Movements",
                 geo=dict(scope="north america", projection_type="mercator")
             )
             st.plotly_chart(fig)

             # Display data table
             st.subheader("Movement Details")
             columns_to_display = [
                 "LS_DRIVER", "LS_POWER_UNIT", "LS_TRAILER1", "LEGO_ZONE_DESC",
                 "LEGD_ZONE_DESC", "LS_TO_ZONE", "LS_LEG_DIST", "LS_MT_LOADED",
                 "INS_TIMESTAMP", "LS_LEG_NOTE"
             ]
             st.dataframe(filtered_data[columns_to_display], use_container_width=True)

 except Exception as e:
     st.error(f"Error processing the file: {e}")
else:
 st.info("Please upload a valid LEGSUM Excel file to proceed.")
