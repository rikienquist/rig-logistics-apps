import pandas as pd
import streamlit as st

# Streamlit App Title and Instructions
st.title("madd_debug Route Processor")

st.markdown("""
### Instructions:
Use the following query to generate the required madd_debug data:  
`select * from madd_debug where int_data='X';`  

Replace X with the desired INT_DATA.  

Save the query results as a CSV file and upload below to see the final route for the data.
""")

# File upload section
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load the CSV file
    df = pd.read_csv(uploaded_file)

    # Parse MESSAGE column to extract relevant data
    def parse_message(message):
        try:
            action, data = message.split(" - ", 1)
            details = dict(item.split(": ") for item in data.split(", "))
            details["ACTION"] = action
            return details
        except Exception as e:
            return None

    # Extract and normalize MESSAGE column
    parsed_messages = df["MESSAGE"].apply(parse_message).dropna().apply(pd.Series)

    # Keep only relevant columns
    parsed_messages["LS_LEG_SEQ"] = parsed_messages["LS_LEG_SEQ"].astype(int)
    parsed_messages["TIMESTAMP"] = pd.to_datetime(parsed_messages["TIMESTAMP"], errors='coerce')

    final_data = parsed_messages[["LS_TRIP_NUMBER", "LS_LEG_SEQ", "LS_FROM_ZONE", "LS_TO_ZONE", "ACTION", "TIMESTAMP"]]

    # Process actions to determine the final sequence
    def process_routes(data):
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
            # Sort sequence by leg and format as a single string
            sorted_legs = [sequence[key] for key in sorted(sequence)]
            postal_code_route = " → ".join([sorted_legs[0][0]] + [leg[1] for leg in sorted_legs])
            results[trip] = postal_code_route
        return results

    final_routes = process_routes(final_data)

    # Display the results
    st.markdown("### Final Routes:")
    for trip, route in final_routes.items():
        st.write(f"Trip {trip}: {route}")
