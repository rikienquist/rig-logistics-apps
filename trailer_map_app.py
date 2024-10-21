import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

trailer_data = pd.read_excel('data/Trailer Counts Sept 15-21.xlsx', sheet_name='attachment')
coordinates_data = pd.read_excel('data/Coordinates.xlsx', sheet_name='Sheet1')

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
        # Calculate trailers leaving and arriving for each terminal
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
            hoverinfo='text' if mode == 'Terminals Only' else 'skip',  # Show terminal info only in 'Terminals Only' mode
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
            scope='north america',  # Set scope to focus on North America
            projection_type='natural earth',  # Use a projection that focuses on this region
            showland=True,
            landcolor="lightgray"
        ),
        height=700,  # Adjusting map size
        showlegend=False,
        margin={"r":0,"t":0,"l":0,"b":0},  # Set margins for fullscreen view
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

# Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Trailer Movement Map"),
    html.Label("Select Mode:"),
    dcc.Dropdown(
        id='mode-dropdown',
        options=[
            {'label': 'Terminals Only', 'value': 'Terminals Only'},
            {'label': 'All Routes', 'value': 'All Routes'}
        ],
        value='Terminals Only'
    ),
    html.Label("Select Class:"),
    dcc.Dropdown(
        id='class-dropdown',
        options=[
            {'label': 'All', 'value': 'ALL'},
            {'label': 'Dry Van', 'value': 'DRY VAN'},
            {'label': 'Single Tem', 'value': 'SINGLE TEM'},
            {'label': 'Tri Temp', 'value': 'TRI TEMP'}
        ],
        value='ALL'
    ),
    html.Label("Select Specific Routes (Optional for 'All Routes')"),
    dcc.Dropdown(
        id='route-dropdown',
        options=[],
        multi=True
    ),
    dcc.Graph(id='map-graph')
])

# Update the map when the dropdowns change
@app.callback(
    [Output('map-graph', 'figure'),
     Output('route-dropdown', 'options')],
    [Input('mode-dropdown', 'value'),
     Input('class-dropdown', 'value'),
     Input('route-dropdown', 'value')]
)
def update_map(selected_mode, selected_class, selected_routes):
    # Filter routes dropdown based on class
    filtered_data = filter_trailers_by_class(trailer_data, selected_class)
    unique_routes = filtered_data.groupby(['ORIGPROV 2', 'DESTPROV 2']).size().reset_index(name='Total Trailers')

    # Remove routes where origin and destination are the same
    unique_routes = unique_routes[unique_routes['ORIGPROV 2'] != unique_routes['DESTPROV 2']]

    route_options = [{'label': f"{row['ORIGPROV 2']} to {row['DESTPROV 2']}", 'value': f"{row['ORIGPROV 2']} to {row['DESTPROV 2']}"} for _, row in unique_routes.iterrows()]
    
    # Create map with selected routes (if mode is All Routes)
    fig = create_trailer_map(trailer_data, coordinates_data, selected_class, selected_mode, selected_routes)
    
    return fig, route_options

if __name__ == '__main__':
    app.run_server(debug=True)
