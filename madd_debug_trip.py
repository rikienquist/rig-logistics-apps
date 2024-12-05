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

    # Include MESSAGE_ID and CREATED in the parsed data
    parsed_messages["MESSAGE_ID"] = df["MESSAGE_ID"]
    # Truncate CREATED to remove seconds
    parsed_messages["CREATED"] = pd.to_datetime(df["CREATED"]).dt.strftime('%Y-%m-%d %H:%M')

    # Prepare the data for the breakdown
    breakdown_columns = ["MESSAGE_ID", "ACTION", "LS_LEG_SEQ", "LS_FROM_ZONE", "LS_TO_ZONE", "USER", "CREATED"]
    breakdown_data = parsed_messages[breakdown_columns]

    # Group data by CREATED timestamp
    grouped_by_created = breakdown_data.groupby("CREATED")

    # Display breakdown by timestamps
    for i, (timestamp, group) in enumerate(grouped_by_created, start=1):
        st.markdown(f"### Timestamp {i}: {timestamp}")
        # Sort the group by MESSAGE_ID
        group = group.sort_values(by="MESSAGE_ID")
        group_display = group.copy()
        group_display["Route"] = group_display["LS_FROM_ZONE"] + " → " + group_display["LS_TO_ZONE"]
        group_display = group_display[["MESSAGE_ID", "ACTION", "LS_LEG_SEQ", "Route", "USER"]]
        st.write(group_display)

    # Initialize the leg sequence processing for final routes
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
                    # Add missing leg if LS_FROM_ZONE doesn't match LS_TO_ZONE
                    unmatched_leg = remaining_legs.pop(0)
                    if unmatched_leg[1][0] != current_to_zone:
                        route.append((current_to_zone, unmatched_leg[1][0]))
                    route.append(unmatched_leg[1])

        # Build the final postal code route
        postal_code_route = " → ".join([route[0][0]] + [leg[1] for leg in route])
        return postal_code_route

    # Process each trip and get the final routes
    parsed_messages["LS_LEG_SEQ"] = parsed_messages["LS_LEG_SEQ"].astype(int)
    processed_routes = parsed_messages.groupby("LS_TRIP_NUMBER").apply(process_routes)

    # Deduplicate trip routes by converting to a dictionary
    final_routes = processed_routes.to_dict()

    # Display the final routes
    st.markdown("### Final Routes:")
    for trip, route in final_routes.items():
        st.write(f"Trip {trip}: {route}")
