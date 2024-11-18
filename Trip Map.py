# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import geodesic
import warnings

warnings.filterwarnings("ignore")

# Streamlit application
st.title("Trip Map Dashboard")

# Load data from GitHub
@st.cache_data
def load_data():
    # Adjust paths to point to your GitHub repository
    tlorder_df = pd.read_csv("trip_map_data/TLORDER_Sep2022-Sep2024_V3.csv", low_memory=False)
    geocode_df = pd.read_csv("trip_map_data/merged_geocoded.csv", low_memory=False)
    driver_pay_df = pd.read_csv("trip_map_data/driver_pay_data.csv", low_memory=False)
    return tlorder_df, geocode_df, driver_pay_df

# Load datasets
tlorder_df, geocode_df, driver_pay_df = load_data()

# Coordinate corrections
coordinate_fixes = {
    ("ACHESON", "AB"): {"LAT": 53.5522, "LON": -113.7627},
    ("BALZAC", "AB"): {"LAT": 51.2126, "LON": -114.0076},
    ("LACOMBE", "AB"): {"LAT": 54.4682, "LON": -113.7305},
    ("RED WATER", "AB"): {"LAT": 53.9520, "LON": -113.1089},
    ("BROOKS", "AB"): {"LAT": 50.5657, "LON": -111.8978},
    ("Hardisty", "AB"): {"LAT": 52.6734, "LON": -111.3075},
    ("Falun", "AB"): {"LAT": 52.9598, "LON": -113.8253},
    ("STETTLER", "AB"): {"LAT": 52.3235, "LON": -112.7192},
    ("MANVILLE", "AB"): {"LAT": 53.3402, "LON": -111.1752},
    ("Thorsby", "AB"): {"LAT": 53.2272, "LON": -114.0480},
    ("BEAUMONT", "AB"): {"LAT": 53.3521, "LON": -113.4151},
    ("BOYLE", "AB"): {"LAT": 54.5873, "LON": -112.8034},
    ("GRAND CACHE", "AB"): {"LAT": 53.8886, "LON": -119.1143},
    ("MILLET", "AB"): {"LAT": 53.0907, "LON": -113.4685},
    ("NELSON", "BC"): {"LAT": 49.4928, "LON": -117.2948},
    ("SURREY", "BC"): {"LAT": 49.1913, "LON": -122.8490},
    ("DELTA", "BC"): {"LAT": 49.0952, "LON": -123.0265},
    ("GRAND FORKS", "BC"): {"LAT": 49.0335, "LON": -118.4389},
    ("CRESTON", "BC"): {"LAT": 49.0955, "LON": -116.5135},
    ("RICHMOND", "BC"): {"LAT": 49.1666, "LON": -123.1336},
    ("BLUMENORT", "MB"): {"LAT": 49.6079, "LON": -96.6935},
    ("ST ANDREWS", "MB"): {"LAT": 50.2625, "LON": -96.9842},
    ("MORDEN", "MB"): {"LAT": 49.1923, "LON": -98.1143},
    ("STEINBACH", "MB"): {"LAT": 49.5303, "LON": -96.6912},
    ("WINKLER", "MB"): {"LAT": 49.1802, "LON": -97.9391},
    ("BOLTON", "ON"): {"LAT": 43.8742, "LON": -79.7307},
    ("LIVELY", "ON"): {"LAT": 46.4366, "LON": -81.1466},
    ("VERNON", "CA"): {"LAT": 34.0039, "LON": -118.2301},
    ("SELMA", "CA"): {"LAT": 36.5708, "LON": -119.6121},
    ("TROY", "NY"): {"LAT": 42.7284, "LON": -73.6918},
    ("MOTLEY", "MN"): {"LAT": 46.3366, "LON": -94.6462},
}

def correct_coordinates(row):
    orig_key = (row['ORIGCITY'], row['ORIGPROV'])
    dest_key = (row['DESTCITY'], row['DESTPROV'])
    if orig_key in coordinate_fixes:
        row['ORIG_LAT'], row['ORIG_LON'] = coordinate_fixes[orig_key].values()
    if dest_key in coordinate_fixes:
        row['DEST_LAT'], row['DEST_LON'] = coordinate_fixes[dest_key].values()
    return row

# Preprocess Data
st.write("Processing data...")
tlorder_df = tlorder_df.merge(
    geocode_df[['ORIGCITY', 'ORIG_LAT', 'ORIG_LON']].drop_duplicates(),
    on='ORIGCITY', how='left'
).merge(
    geocode_df[['DESTCITY', 'DEST_LAT', 'DEST_LON']].drop_duplicates(),
    on='DESTCITY', how='left'
).apply(correct_coordinates, axis=1)

tlorder_df = tlorder_df[tlorder_df['ORIGCITY'] != tlorder_df['DESTCITY']]

driver_pay_agg = driver_pay_df.groupby('BILL_NUMBER').agg({'TOTAL_PAY_AMT': 'sum', 'DRIVER_ID': 'first'})
tlorder_df = tlorder_df.merge(driver_pay_agg, on='BILL_NUMBER', how='left')

tlorder_df['TOTAL_CHARGE_CAD'] = tlorder_df.apply(
    lambda x: (x['CHARGES'] + x['XCHARGES']) * 1.38 if x['CURRENCY_CODE'] == 'USD' else x['CHARGES'] + x['XCHARGES'], 
    axis=1
)
filtered_df = tlorder_df[(tlorder_df['TOTAL_CHARGE_CAD'] != 0) & (tlorder_df['DISTANCE'] != 0)]
filtered_df.dropna(subset=['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON'], inplace=True)

filtered_df['Revenue per Mile'] = filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['DISTANCE']
filtered_df['Profit Margin (%)'] = (filtered_df['TOTAL_CHARGE_CAD'] / filtered_df['TOTAL_PAY_AMT']) * 100

filtered_df['route_key'] = filtered_df[['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON']].apply(tuple, axis=1)
unique_routes = filtered_df.drop_duplicates(subset='route_key')
unique_routes['Geopy_Distance'] = unique_routes['route_key'].apply(
    lambda r: geodesic((r[0], r[1]), (r[2], r[3])).miles
)
filtered_df = filtered_df.merge(unique_routes[['route_key', 'Geopy_Distance']], on='route_key', how='left')

# Sidebar for inputs
st.sidebar.title("Filters")
selected_punit = st.sidebar.selectbox("Select PUNIT", ["All"] + list(filtered_df['PICK_UP_PUNIT'].unique()))
selected_driver_id = st.sidebar.selectbox("Select Driver ID", ["All"] + list(filtered_df['DRIVER_ID'].dropna().unique()))

# Filter data
data = filtered_df
if selected_punit != "All":
    data = data[data['PICK_UP_PUNIT'] == selected_punit]
if selected_driver_id != "All":
    data = data[data['DRIVER_ID'] == selected_driver_id]

# Display a map with routes
if not data.empty:
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row['ORIG_LON'], row['DEST_LON']],
            lat=[row['ORIG_LAT'], row['DEST_LAT']],
            mode='lines+markers',
            line=dict(width=2, color='blue'),
            marker=dict(size=5),
            name=f"{row['ORIGCITY']} to {row['DESTCITY']}",
            hovertext=(
                f"Route: {row['ORIGCITY']}, {row['ORIGPROV']} → {row['DESTCITY']}, {row['DESTPROV']}<br>"
                f"Revenue per Mile: ${row['Revenue per Mile']:.2f}<br>"
                f"Profit Margin: {row['Profit Margin (%)']:.2f}%"
            )
        ))
    fig.update_layout(
        geo=dict(
            scope='north america',
            projection_type='mercator'
        )
    )
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected filters.")
