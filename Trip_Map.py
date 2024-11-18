import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from geopy.distance import geodesic
import warnings

warnings.filterwarnings("ignore")

# Load datasets
@st.cache
def load_data():
    tlorder_df = pd.read_csv("trip_map_data/TLORDER_Sep2022-Sep2024_V3.csv", low_memory=False)
    geocode_df = pd.read_csv("trip_map_data/merged_geocoded.csv", low_memory=False)
    driver_pay_df = pd.read_csv("trip_map_data/driver_pay_data.csv", low_memory=False)
    return tlorder_df, geocode_df, driver_pay_df

# Load the data
tlorder_df, geocode_df, driver_pay_df = load_data()

# Coordinate fixes
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

# Correct coordinates
def correct_coordinates(row):
    orig_key = (row['ORIGCITY'], row['ORIGPROV'])
    dest_key = (row['DESTCITY'], row['DESTPROV'])
    if orig_key in coordinate_fixes:
        row['ORIG_LAT'], row['ORIG_LON'] = coordinate_fixes[orig_key].values()
    if dest_key in coordinate_fixes:
        row['DEST_LAT'], row['DEST_LON'] = coordinate_fixes[dest_key].values()
    return row

# Preprocess data
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

# Geopy distances
filtered_df['route_key'] = filtered_df[['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON']].apply(tuple, axis=1)
unique_routes = filtered_df.drop_duplicates(subset='route_key')
unique_routes['Geopy_Distance'] = unique_routes['route_key'].apply(
    lambda r: geodesic((r[0], r[1]), (r[2], r[3])).miles
)
filtered_df = filtered_df.merge(unique_routes[['route_key', 'Geopy_Distance']], on='route_key', how='left')

# Streamlit app
st.title("Trip Map Visualization")

punit_options = sorted(filtered_df['PICK_UP_PUNIT'].unique())
punit = st.selectbox("Select PUNIT", options=["All"] + punit_options)

if punit != "All":
    data = filtered_df[filtered_df['PICK_UP_PUNIT'] == punit]
else:
    data = filtered_df

driver_options = sorted(data['DRIVER_ID'].unique())
driver = st.selectbox("Select Driver ID (Optional)", options=["All"] + driver_options)

if driver != "All":
    data = data[data['DRIVER_ID'] == driver]

data = data.sort_values(by='PICK_UP_BY')
daily_routes = list(data.groupby(data['PICK_UP_BY']))
day_index = st.session_state.get("day_index", 0)

total_days = len(daily_routes)
st.write(f"Day {day_index + 1} of {total_days}")

if st.button("Previous Day") and day_index > 0:
    day_index -= 1
    st.session_state["day_index"] = day_index

if st.button("Next Day") and day_index < total_days - 1:
    day_index += 1
    st.session_state["day_index"] = day_index

day, day_data = daily_routes[day_index]

fig = go.Figure()
route_summary = []

for _, row in day_data.iterrows():
    route_summary.append({
        "Route": f"{row['ORIGCITY']} to {row['DESTCITY']}",
        "Total Charge (CAD)": row['TOTAL_CHARGE_CAD'],
        "Distance (miles)": row['DISTANCE'],
        "Revenue per Mile": row['Revenue per Mile'],
        "Profit Margin (%)": row['Profit Margin (%)']
    })

    fig.add_trace(go.Scattergeo(
        lon=[row['ORIG_LON'], row['DEST_LON']],
        lat=[row['ORIG_LAT'], row['DEST_LAT']],
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2),
        name=f"{row['ORIGCITY']} to {row['DESTCITY']}"
    ))

st.plotly_chart(fig)

st.write(pd.DataFrame(route_summary))
