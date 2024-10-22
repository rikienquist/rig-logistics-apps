import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Load static coordinates data
coordinates_data = pd.read_excel('trailer_count_data/Coordinates.xlsx', sheet_name='Sheet1')

# Streamlit app
st.title("Trailer Movement Map")

# File uploader for trailer data
uploaded_trailer_file = st.file_uploader("Upload Trailer Count File (make sure the sheet name is 'attachment'", type=['xlsx'])

# Load trailer data if a file is uploaded, otherwise show a message
if uploaded_trailer_file:
    trailer_data = pd.read_excel(uploaded_trailer_file, sheet_name='attachment')
else:
    st.warning("Please upload a Trailer Count Excel file to visualize the data.")
    st.stop()  # Stop the script until a file is uploaded

# Filter the trailers based on the selected class
def filter_trailers_by_class(trailer_data, selected_class):
    if selected_class == 'ALL':
        return trailer_data
    return trailer_data[trailer_data['CLASS'] == selected_class]

# Define the function to create the map
def create_trailer_map(trailer_data, coordinates_data, selected_class, mode, selected_routes=None):
    fig = go.Figure()

    # Filter the trailers based on the selected class
    filtered_data = filter_trailers_by_class(trailer_data, selected_class)
    
    # Plot terminal points - Keep terminal circles always visible
    for _, row in coordinates_data.iterrows():
        terminal_name = row['ORIGPROV 2']
        trailers_leaving = filtered_data[(filtered_data['ORIGPROV 2'] == terminal_name) & (filtered_data['DESTPROV 2'] != terminal_name)]['ORIGPROV 2'].count()
        trailers_arriving = filtered_data[(filtered_data['DESTPROV 2'] == terminal_name) & (filtered_data['ORIGPROV 2'] != terminal_name)]['DESTPROV 2'].count()
        net_difference = trailers_arriving - trailers_leaving
        
        # Plot the terminal circle (Visible in both modes)
        fig.add_trace(go.Scattergeo(
            locationmode='USA-states',
            lon=[row['Longitude']],
            lat=[row['Latitude']],
            text=f"Terminal: {terminal_name}<br>Total Trailers Leaving: {trailers_leaving}<br>Total Trailers Arriving: {trailers_arriving}<br>Total Net Trailer Difference: {net_difference}",
            marker=dict(size=10, color='blue'),
            name=f'Terminal: {terminal_name}',
            hoverinfo='text' if mode == 'Terminals Only' else 'skip',
        ))

    # Plot routes if "All Routes" is selected
    if mode == 'All Routes':
        unique_routes = filtered_data.groupby(['ORIGPROV 2', 'DESTPROV 2']).size().reset_index(name='Total Trailers')

        # Remove routes where the origin and destination are the same (e.g., AB to AB)
        unique_routes = unique_routes[unique_routes['ORIGPROV 2'] != unique_routes['DESTPROV 2']]

        # If a specific route is selected, filter the routes
        if selected_routes:
            unique_routes = unique_routes[unique_routes.apply(lambda x: f"{x['ORIGPROV 2']} to {x['DESTPROV 2']}" in selected_routes, axis=1)]

        # Plot each route
        for _, route in unique_routes.iterrows():
            origin = coordinates_data[coordinates_data['ORIGPROV 2'] == route['ORIGPROV 2']].iloc[0]
            destination = coordinates_data[coordinates_data['ORIGPROV 2'] == route['DESTPROV 2']].iloc[0]

            ab_to_mb = filtered_data[(filtered_data['ORIGPROV 2'] == route['ORIGPROV 2']) & (filtered_data['DESTPROV 2'] == route['DESTPROV 2'])].shape[0]
            mb_to_ab = filtered_data[(filtered_data['ORIGPROV 2'] == route['DESTPROV 2']) & (filtered_data['DESTPROV 2'] == route['ORIGPROV 2'])].shape[0]
            ab_net = mb_to_ab - ab_to_mb
            mb_net = ab_to_mb - mb_to_ab

            # Add route lines
            fig.add_trace(go.Scattergeo(
                locationmode='USA-states',
                lon=[origin['Longitude'], destination['Longitude']],
                lat=[origin['Latitude'], destination['Latitude']],
                mode='lines',
                line=dict(width=2, color='red'),
                text=f"{route['ORIGPROV 2']} to {route['DESTPROV 2']}: {ab_to_mb}<br>{route['DESTPROV 2']} to {route['ORIGPROV 2']}: {mb_to_ab}<br>Net Difference for {route['ORIGPROV 2']}: {ab_net}<br>Net Difference for {route['DESTPROV 2']}: {mb_net}",
                hoverinfo='text',
                name=f"{route['ORIGPROV 2']} to {route['DESTPROV 2']}"
            ))

    # Ensure map layout and projection stay consistent
    fig.update_layout(
        title=f"Trailer Movement Map (Filter by CLASS: {selected_class}, Mode: {mode})",
        geo=dict(
            scope='north america',
            projection_type='natural earth',
            showland=True,
            landcolor="lightgray"
        ),
        height=700,
        showlegend=False,
        margin={"r":0,"t":0,"l":0,"b":0},
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=[dict(
                args=["mode", "fullscreen"],
                label="Full Screen",
                method="relayout"
            )],
            pad={"r": 10, "t": 10},
            showactive=False,
            x=1,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )]  # Fullscreen button
    )
    return fig

# Streamlit app controls
selected_mode = st.selectbox("Select Mode", ['Terminals Only', 'All Routes'])
selected_class = st.selectbox("Select Class", ['ALL', 'DRY VAN', 'SINGLE TEM', 'TRI TEMP'])

if selected_mode == 'All Routes':
    # Allow route selection only if "All Routes" is selected
    filtered_data = filter_trailers_by_class(trailer_data, selected_class)
    unique_routes = filtered_data.groupby(['ORIGPROV 2', 'DESTPROV 2']).size().reset_index(name='Total Trailers')
    
    # Remove routes where origin and destination are the same
    unique_routes = unique_routes[unique_routes['ORIGPROV 2'] != unique_routes['DESTPROV 2']]
    
    route_options = [f"{row['ORIGPROV 2']} to {row['DESTPROV 2']}" for _, row in unique_routes.iterrows()]
    selected_routes = st.multiselect("Select Specific Routes (Optional)", route_options)
else:
    selected_routes = None

# Create and display the map
fig = create_trailer_map(trailer_data, coordinates_data, selected_class, selected_mode, selected_routes)
st.plotly_chart(fig)
