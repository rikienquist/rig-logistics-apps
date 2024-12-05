import pandas as pd
import streamlit as st

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

    # Ensure CREATED is a datetime column and round to minutes (removing seconds)
    df["CREATED"] = pd.to_datetime(df["CREATED"]).dt.strftime("%Y-%m-%d %H:%M")

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

    # Include relevant columns
    parsed_messages["MESSAGE_ID"] = df["MESSAGE_ID"]
    parsed_messages["CREATED"] = df["CREATED"]
    parsed_messages["LS_LEG_SEQ"] = parsed_messages["LS_LEG_SEQ"].astype(int)

    # Sort the data by MESSAGE_ID to ensure operations are applied in order
    parsed_messages = parsed_messages.sort_values(by=["CREATED", "MESSAGE_ID"])

    # Generate breakdown tables for each unique CREATED timestamp
    st.markdown("### Breakdown by Timestamps")
    for timestamp in parsed_messages["CREATED"].unique():
        st.markdown(f"#### Timestamp: {timestamp}")

        # Filter data for the current timestamp
        breakdown_data = parsed_messages[parsed_messages["CREATED"] == timestamp]

        # Prepare breakdown table
        breakdown_table = breakdown_data[["MESSAGE_ID", "ACTION", "LS_LEG_SEQ", "LS_FROM_ZONE", "LS_TO_ZONE", "USER"]]
        breakdown_table["Route"] = breakdown_table["LS_FROM_ZONE"] + " → " + breakdown_table["LS_TO_ZONE"]
        breakdown_table["Action"] = breakdown_table["ACTION"].apply(
            lambda x: f"**:green[{x}]**" if x == "INSERT" else f"**:red[{x}]**"
        )

        # Display the breakdown table
        st.table(breakdown_table[["MESSAGE_ID", "Action", "LS_LEG_SEQ", "Route", "USER"]])

    # Initialize the leg sequence processing
    def process_routes(data):
        sequence = {}  # Store the leg sequences as a dictionary
        for _, row in data.iterrows():
            leg_seq = row["LS_LEG_SEQ"]
            action = row["ACTION"]
            from_zone = row["LS_FROM_ZONE"]
            to_zone = row["LS_TO_ZONE"]

            if action == "INSERT":
                sequence[leg_seq] = (from_zone, to_zone)
            elif action == "DELETE":
                if leg_seq in sequence:
                    del sequence[leg_seq]

        # Sort the final sequence by leg order
        sorted_legs = [(k, sequence[k]) for k in sorted(sequence)]

        # Reorder legs into a continuous route
        route = []
        remaining_legs = sorted_legs.copy()

        while remaining_legs:
            if not route:
                # Begin with the first leg
                route.append(remaining_legs.pop(0)[1])
            else:
                current_to_zone = route[-1][1]
                # Find the next leg where LS_FROM_ZONE matches the current LS_TO_ZONE
                next_leg = next(
                    (leg for leg in remaining_legs if leg[1][0] == current_to_zone),
                    None,
                )

                if next_leg:
                    route.append(next_leg[1])
                    remaining_legs.remove(next_leg)
                else:
                    # If no match is found, pick the next available leg
                    unmatched_leg = remaining_legs.pop(0)
                    # Insert missing `LS_FROM_ZONE` if it's not connected
                    if unmatched_leg[1][0] != current_to_zone:
                        route.append((current_to_zone, unmatched_leg[1][0]))
                    # Add the unmatched leg
                    route.append(unmatched_leg[1])

        # Build the final postal code route
        postal_code_route = " → ".join([route[0][0]] + [leg[1] for leg in route])
        return postal_code_route

    # Process each trip and get the final routes
    processed_routes = parsed_messages.groupby("LS_TRIP_NUMBER").apply(process_routes)

    # Deduplicate trip routes by converting to a dictionary
    final_routes = processed_routes.to_dict()

    # Display the results
    st.markdown("### Final Routes:")
    for trip, route in final_routes.items():
        st.write(f"Trip {trip}: {route}")
