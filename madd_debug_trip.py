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

    # Include MESSAGE_ID to sort operations in correct order
    parsed_messages["MESSAGE_ID"] = df["MESSAGE_ID"]
    parsed_messages["LS_LEG_SEQ"] = parsed_messages["LS_LEG_SEQ"].astype(int)

    # Sort the data by MESSAGE_ID to simulate the order of operations
    parsed_messages = parsed_messages.sort_values(by="MESSAGE_ID")

    # Initialize the leg sequence processing
    def process_routes(data):
        sequence = {}  # Store the leg sequences as a dictionary
        for _, row in data.iterrows():
            leg_seq = row["LS_LEG_SEQ"]
            action = row["ACTION"]
            from_zone = row["LS_FROM_ZONE"]
            to_zone = row["LS_TO_ZONE"]

            if action == "INSERT":
                # Shift existing legs back if necessary
                temp_sequence = {}
                for existing_seq in sorted(sequence.keys(), reverse=True):
                    if existing_seq >= leg_seq:
                        temp_sequence[existing_seq + 1] = sequence[existing_seq]
                    else:
                        temp_sequence[existing_seq] = sequence[existing_seq]
                temp_sequence[leg_seq] = (from_zone, to_zone)
                sequence = temp_sequence

            elif action == "DELETE":
                # Remove the specified leg sequence
                if leg_seq in sequence:
                    del sequence[leg_seq]

        # Sort the final sequence by leg order
        sorted_legs = [sequence[key] for key in sorted(sequence)]
        postal_code_route = " → ".join([sorted_legs[0][0]] + [leg[1] for leg in sorted_legs])
        return postal_code_route

    # Process each trip and get the final routes
    final_routes = parsed_messages.groupby("LS_TRIP_NUMBER").apply(process_routes)

    # Display the results
    st.markdown("### Final Routes:")
    for trip, route in final_routes.items():
        st.write(f"Trip {trip}: {route}")
