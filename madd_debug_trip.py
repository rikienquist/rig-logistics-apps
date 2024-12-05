import pandas as pd
import streamlit as st
import numpy as np

# Streamlit App Title and Instructions
st.title("madd_debug Route Processor")

st.markdown("""
### Instructions:
Use the following query to generate the required madd_debug data:  
`select * from madd_debug where int_data='X';`  

Replace X with the desired INT_DATA.  

Save the query results as CSV files and upload them below to see the data.
""")

# File upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Parse MESSAGE column to extract relevant data
    def parse_message(row):
        action, data = row.split(" - ", 1)
        details = dict(item.split(": ") for item in data.split(", "))
        details["ACTION"] = action
        return details

    # Extract and normalize MESSAGE column
    parsed_messages = df["MESSAGE"].apply(parse_message).apply(pd.Series)
    df = pd.concat([df, parsed_messages], axis=1)
    
    # Convert numeric columns for sorting
    df["LS_LEG_SEQ"] = df["LS_LEG_SEQ"].astype(int)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors='coerce')
    
    # Filter only relevant columns
    final_data = df[["LS_TRIP_NUMBER", "LS_LEG_SEQ", "LS_FROM_ZONE", "LS_TO_ZONE", "ACTION"]]
    
    # Process actions to determine the final sequence
    def process_routes(data):
        # Group by trip number
        results = {}
        grouped = data.groupby("LS_TRIP_NUMBER")
        for trip, group in grouped:
            group = group.sort_values(by=["TIMESTAMP", "LS_LEG_SEQ"])
            sequence = {}
            for _, row in group.iterrows():
                if row["ACTION"] == "INSERT":
                    sequence[row["LS_LEG_SEQ"]] = (row["LS_FROM_ZONE"], row["LS_TO_ZONE"])
                elif row["ACTION"] == "DELETE" and row["LS_LEG_SEQ"] in sequence:
                    del sequence[row["LS_LEG_SEQ"]]
            # Sort sequence by leg and format
            final_sequence = [sequence[key] for key in sorted(sequence)]
            results[trip] = final_sequence
        return results

    final_routes = process_routes(final_data)
    
    # Display the results
    st.markdown("### Final Routes:")
    for trip, route in final_routes.items():
        st.write(f"Trip {trip}:")
        for i, (from_zone, to_zone) in enumerate(route, start=1):
            st.write(f"  Leg {i}: {from_zone} -> {to_zone}")
