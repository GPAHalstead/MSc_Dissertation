# Calculating Percentage Difference in OPEX between 2 refrigerants on differing tariffs

# Importing libraries
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from scipy.interpolate import RBFInterpolator

# Defining Heat Pump and Tariff data file names
File = 'Heat_Pump_Data_F.xlsx'
Tariff_File = 'Tariff_Data.xlsx'



# User defined Variables - Heat Pump 1 on Tariff 1 vs Heat Pump 2 on Tariff 2
Heat_Pump_1 = 'P-AHP-32-9'
Tariff_1    = 'Economy_7'
Heat_Pump_2 = 'P-AHP-290-9'
Tariff_2    = 'Economy_7'

# User defined Test Conditions
T_cutoff = 16 # Cut-off temperature in degrees celcius
Q_design = 9 # Design Heat Capacity in kW
Resolution = 0.5 # Geospatial calculation resolution in degrees
Op_Mode    = 'On_Off_Cd_0.9'              
Eff_SH     = 0.99   
    
Ref_Tariff_Map = {
    Heat_Pump_1 : Tariff_1,
    Heat_Pump_2 : Tariff_2 
}
Refs = list(Ref_Tariff_Map.keys())

# Quasi-Static Heat Pump Model
# Full load Heat Capacity function
def Q_FL(T_OA, T_LW):
    T_LW = max(min(T_LW, T_LW_max), T_LW_min)
    return float(Capacity(np.array([[T_OA, T_LW]])).item())

# Full load Power consumption function
def P_FL(T_OA, T_LW):
    T_LW = max(min(T_LW, T_LW_max), T_LW_min)
    return float(Power(np.array([[T_OA, T_LW]])).item())

# Total Power Consumption
# Calculates instantaneous electrical input (W) at a given OAT using SPLCF for part-load and backup heater when needed
def PI(T_OA, T_bivalent, Op_Mode, Eff_SH):
    # Weather-compensated LWT curve with cap at 55°C
    T_LW = min(55 + (-30 / (16 - T_bivalent)) * (T_OA - T_bivalent),55)
    if T_OA_min <= T_OA <= T_cutoff:
        # Requested part-load ratio from design load and climate demand share
        PLR = (Q_design / Q_FL(T_OA, T_LW)) * ((T_cutoff - T_OA) / (T_cutoff - T_bivalent))
        if 0 < PLR <= 1:
            # COP at full load and part-load degradation factor
            COP_FL = Q_FL(T_OA, T_LW) / P_FL(T_OA, T_LW)
            Cd = np.interp(PLR, SPLCF['Part_Load'], SPLCF[Op_Mode])
            return (Q_FL(T_OA, T_LW) * PLR) / (Cd * COP_FL)
        elif PLR > 1:
            # Deficit covered by space heater (Eff_SH), plus compressor at full load
            return ((PLR - 1) * Q_FL(T_OA, T_LW) / Eff_SH) + P_FL(T_OA, T_LW)
        else:
            # No demand
            return 0
    elif T_OA < T_OA_min:
        # Below modelable OAT, assume bivalent coverage via SH for the required fraction
        PLR = (Q_design / Q_FL(T_OA, T_LW)) * ((T_cutoff - T_OA) / (T_cutoff - T_bivalent))
        return PLR * Q_FL(T_bivalent, T_LW) / Eff_SH
    else:
        # Above cutoff, no space heating
        return 0

# Load Temperature Datasets by month
Months = ['OCT_2024','NOV_2024','DEC_2024','JAN_2025','FEB_2025','MAR_2025','APR_2025']
Temperature_Datasets = [Dataset(f'UK_T_{m}.nc') for m in Months]

# Generates Array of Latitudes and Longtitudes for which there are temperature values
nc_lats = Temperature_Datasets[0].variables['latitude'][:]
nc_lons = Temperature_Datasets[0].variables['longitude'][:]

# Build array of hour-of-day indices for every timestep across all files (0–23)
Hours = []
for ds in Temperature_Datasets:
    n = ds.variables['t2m'].shape[0]
    arr = (np.arange(n) % 24).astype(int)
    Hours.append(arr)
Hours = np.concatenate(Hours)

# Load UK boundary and prepare geometry for point-in-polygon tests
uk = gpd.read_file("united_kingdom.geojson").to_crs(4326)
uk_shape = uk.unary_union.buffer(0)

# Build coarse grid over UK bounding box based on user-set Resolution
step = Resolution
grid_lats = np.arange(int(np.floor(uk_shape.bounds[1])), int(np.ceil(uk_shape.bounds[3])) + step, step)
grid_lons = np.arange(int(np.floor(uk_shape.bounds[0])), int(np.ceil(uk_shape.bounds[2])) + step, step)

# Filter grid points to only those inside the UK polygon and with non-missing temperature data
valid_points = []
for lat in grid_lats:
    for lon in grid_lons:
        pt = Point(lon, lat)
        if not uk_shape.contains(pt): continue
        lat_idx = np.abs(nc_lats - lat).argmin()
        lon_idx = np.abs(nc_lons - lon).argmin()
        ts = Temperature_Datasets[0].variables['t2m'][:, lat_idx, lon_idx]
        if np.ma.is_masked(ts) and ts.mask.any(): continue
        valid_points.append((lat, lon))

# Load SPLCF table for part-load degradation factors (columns keyed by operating mode)
SPLCF = pd.read_excel(File, sheet_name='Standard_PL_Correction_Factors')

# Load performance maps for each refrigerant: build RBF interpolators for capacity and power
refrigerant_data = {}
for ref in Refs:
    df = pd.read_excel(File, sheet_name=ref)
    df[['OAT','LWT','HC','IP']] = df[['OAT','LWT','HC','IP']].apply(pd.to_numeric, errors='coerce')
    Capacity_map = RBFInterpolator(np.column_stack((df['OAT'], df['LWT'])), df['HC'])
    Power_map = RBFInterpolator(np.column_stack((df['OAT'], df['LWT'])), df['IP'])
    refrigerant_data[ref] = {
        'DSV': df,
        'Capacity': Capacity_map,
        'Power': Power_map,
        'T_OA_min': df['OAT_MIN'].iloc[0],
        'T_LW_min': df['LWT_MIN'].iloc[0],
        'T_LW_max': df['LWT_MAX'].iloc[0]
    }

# Load DNO licence areas and spatially join grid points to the corresponding DNO ID
dno = gpd.read_file("DNO_Licence_Area.geojson").to_crs(4326)
points_gdf = gpd.GeoDataFrame({'idx': range(len(valid_points))}, geometry=[Point(lon, lat) for lat, lon in valid_points], crs=4326)
joined = gpd.sjoin(points_gdf, dno[['ID', 'geometry']], how='left', predicate='within')
dno_ids = joined.sort_values('idx')['ID'].to_numpy()

# Helper - normalise tariff column headers to "HH:MM" strings if Excel stores times as datetime
def _col_to_str(c):
    import datetime
    return f"{c.hour:02d}:{c.minute:02d}" if isinstance(c, datetime.time) else str(c)

# Load tariff matrices for each required tariff sheet, build dict: {DNO_ID: np.array(24h prices)}
Tariff_Data_All = {}
for sheet in set(Ref_Tariff_Map.values()):
    df = pd.read_excel(Tariff_File, sheet_name=sheet)
    df.columns = [_col_to_str(c) for c in df.columns]
    df = df[['ID'] + [f"{h:02d}:00" for h in range(24)]]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    Tariff_Data_All[sheet] = {
        int(row['ID']): row[1:].to_numpy(dtype=float)
        for _, row in df.iterrows()
    }

# Utility - percentage difference relative to the smaller baseline (symmetric % difference)
def pct_diff(a, b):
    d = min(a, b)
    return np.nan if d == 0 or np.isnan(d) else ((a - b) / d) * 100

# Accumulators for geospatial outputs: % difference in seasonal OPEX and energy
Percent_OPEX_Diff = []
Percent_Power_Diff = []

# Main geospatial loop - iterate over valid grid points and compute seasonal metrics
for idx, (lat, lon) in enumerate(valid_points):
    # Lightweight progress printout every ~10% of points
    if idx % max(1, len(valid_points) // 10) == 0:
        percent = (idx / len(valid_points)) * 100
        print(f"🔄 OPEX calc progress: {percent:.1f}% done")

    # Skip points without a mapped DNO
    dno_id = dno_ids[idx]
    if pd.isna(dno_id): 
        Percent_OPEX_Diff.append(np.nan)
        Percent_Power_Diff.append(np.nan)
        continue
    dno_id = int(dno_id)

    # Extract local temperature time series and convert to °C
    lat_idx = np.abs(nc_lats - lat).argmin()
    lon_idx = np.abs(nc_lons - lon).argmin()
    T_series = np.concatenate([ds.variables['t2m'][:, lat_idx, lon_idx].filled(np.nan) for ds in Temperature_Datasets])
    T_C = T_series - 273.15

    # Per-refrigerant seasonal totals
    Opex, Energy = [], []

    for ref in Refs:
        # Fetch hourly tariff vector for the point's DNO under this ref's tariff
        tariff_name = Ref_Tariff_Map[ref]
        tariffs = Tariff_Data_All[tariff_name]
        if dno_id not in tariffs:
            Opex.append(np.nan)
            Energy.append(np.nan)
            continue

        hourly_tariff = tariffs[dno_id]

        # Bind performance maps and bounds for the selected refrigerant
        rd = refrigerant_data[ref]
        Capacity, Power = rd['Capacity'], rd['Power']
        T_OA_min = rd['T_OA_min']
        T_LW_min, T_LW_max = rd['T_LW_min'], rd['T_LW_max']

        # Bivalent OAT from local climate (5th percentile of OAT distribution)
        T_Bivalent = np.nanpercentile(T_series, 5) - 273.15

        # Expose variables used inside PI via globals() to match existing function signature
        globals().update(locals())

        # Aggregate seasonal energy and OPEX by stepping through each hour
        opex = energy = 0
        for T, hr in zip(T_C, Hours):
            if np.isnan(T): continue
            p = PI(T, T_Bivalent, Op_Mode, Eff_SH)
            energy += p
            opex += p * hourly_tariff[hr]

        # Convert pence to £ and append to lists
        Opex.append(opex / 100)
        Energy.append(energy)

    # Store symmetric % differences between the two refrigerants for this location
    Percent_OPEX_Diff.append(pct_diff(*Opex))
    Percent_Power_Diff.append(pct_diff(*Energy))

# Assemble geodataframe of point results for mapping and export
gdf = gpd.GeoDataFrame({
    'geometry': [Point(lon, lat) for lat, lon in valid_points],
    'Percent_OPEX_Difference': Percent_OPEX_Diff,
    'Percent_Power_Difference': Percent_Power_Diff
}, geometry='geometry', crs='EPSG:4326')

# Plotting - quick-look choropleths for % OPEX and % Energy differences
for col, title in [('Percent_OPEX_Difference', 'OPEX'), ('Percent_Power_Difference', 'Power')]:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=col, cmap='coolwarm', markersize=200, legend=True, ax=ax)
    gpd.GeoDataFrame(geometry=[uk_shape]).boundary.plot(ax=ax, color='black')
    plt.title(f'Percentage Difference in {title} by Refrigerant (%)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Save
# Persist results to GeoJSON with filename describing the comparison and tariffs
#tag = f"{Heat_Pump_1}_{Tariff_1}_vs_{Heat_Pump_2}_{Tariff_2}"
#gdf.to_file(f"{tag}.geojson", driver='GeoJSON')
