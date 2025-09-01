# Linear, Thin-plate Spline and Cubic Interpolation Techniques

import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator

# Define interpolation and performance functions
def HC_FL(OAT, LWT):
    return Capacity(np.array([[OAT, LWT]]))

def PI_FL(OAT, LWT):
    return Power(np.array([[OAT, LWT]]))

def PI(OAT, LWT, HC, Op_Mode, Eff_SH):
    PLR = HC / HC_FL(OAT, LWT)
    COP_FL = HC_FL(OAT, LWT) / PI_FL(OAT, LWT)
    CF = np.interp(PLR, SPLCF['Part_Load'], SPLCF[Op_Mode])
    return (HC_FL(OAT, LWT) * PLR) / (CF * COP_FL)

SPLCF = pd.read_excel('Heat_Pump_Data_F.xlsx', sheet_name='Standard_PL_Correction_Factors')

# Define the column headings for comparison DataFrame
columns = [
    "Interpolator",
    "Average_Percentage_OP_Error",
    "Detached_OP_Error",
    "End-Terrace_OP_Error",
    "Semi-Detached_OP_Error",
    "Mid-Terrace_OP_Error",
    "Detached_HS_Accuracy",
    "End-Terrace_HS_Accuracy",
    "Semi-Detached_HS_Accuracy",
    "Mid-Terrace_HS_Accuracy",
    "5_OP_Error",
    "8_OP_Error",
    "11_OP_Error",
    "5_HS_Accuracy",
    "8_HS_Accuracy",
    "11_HS_Accuracy",
]

# Create empty DataFrame for storing results
Comparison = pd.DataFrame(columns=columns)

# List of interpolation kernels to compare
Interpolation_Method_List = ['linear','thin_plate_spline','cubic']

for Interpolator in Interpolation_Method_List:
    # Load property metadata
    DB_Properties = pd.read_csv(
        'Property_Data_Summary.csv',
        usecols=[
            'Property_ID', 'Excluded_SPF_analysis_reason', 'House_Form', 'HP_Model',
            'Cleansed_dataset_start', 'Cleansed_dataset_end'
        ]
    )
    DB_Properties['Cleansed_dataset_start'] = pd.to_datetime(
        DB_Properties['Cleansed_dataset_start'], errors='coerce', utc=True
    )
    DB_Properties['Cleansed_dataset_end'] = pd.to_datetime(
        DB_Properties['Cleansed_dataset_end'], errors='coerce', utc=True
    )
    
    # Heat pump models and heating periods
    List_HP_Models = [
        'PUHZ-W50VHA2', 'PUZ-WM85VAA',
        'PUZ-WM112VAA'
    ]
    heating_periods = [
        (pd.Timestamp('2020-10-01', tz='UTC'), pd.Timestamp('2021-04-01', tz='UTC')),
        (pd.Timestamp('2021-10-01', tz='UTC'), pd.Timestamp('2022-04-01', tz='UTC')),
        (pd.Timestamp('2022-10-01', tz='UTC'), pd.Timestamp('2023-04-01', tz='UTC'))
    ]
    
    # Filter properties by coverage, quality, model, and house form
    DB_Properties_Filtered = DB_Properties.loc[
        (DB_Properties['Cleansed_dataset_start'] <= heating_periods[-1][1]) &
        (DB_Properties['Cleansed_dataset_end'] >= heating_periods[0][0]) &
        (DB_Properties['Excluded_SPF_analysis_reason'] != 'Quality score outside threshold') &
        (DB_Properties['HP_Model'].isin(List_HP_Models)) &
        (DB_Properties['House_Form'].isin([
            'Detached','End-Terrace','Semi-Detached','Mid-Terrace'
        ]))
    ].reset_index(drop=True)

    # Prepare for processing
    total_props = DB_Properties_Filtered.shape[0]
    counter = 0
    print(f"Starting processing of {total_props} properties across multiple heating seasons...")
    
    # Load half-hourly data in chunks and filter to heating periods and selected properties
    all_props = DB_Properties_Filtered['Property_ID'].tolist()
    chunks = pd.read_csv(
        'eoh_cleaned_half_hourly.csv',
        usecols=[
            'Timestamp','Property_ID','External_Air_Temperature',
            'Heat_Pump_Energy_Output','Whole_System_Energy_Consumed',
            'Heat_Pump_Heating_Flow_Temperature'
        ],
        parse_dates=['Timestamp'],
        chunksize=200_000
    )
    filtered = []
    for chunk in chunks:
        chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], utc=True)
        mask = np.zeros(len(chunk), dtype=bool)
        for start, end in heating_periods:
            mask |= (chunk['Timestamp'] >= start) & (chunk['Timestamp'] <= end)
        mask &= chunk['Property_ID'].isin(all_props)
        filtered.append(chunk.loc[mask])
    
    df_eoh = pd.concat(filtered, ignore_index=True)

    # Clean and rename columns
    df_eoh = (
        df_eoh.dropna(subset=[
            'External_Air_Temperature','Heat_Pump_Energy_Output',
            'Whole_System_Energy_Consumed','Heat_Pump_Heating_Flow_Temperature'
        ])
        .rename(columns={
            'External_Air_Temperature':'OAT',
            'Heat_Pump_Energy_Output':'Q',
            'Whole_System_Energy_Consumed':'PI',
            'Heat_Pump_Heating_Flow_Temperature':'LWT'
        })
    )
    
    # Group by property
    grouped = df_eoh.groupby('Property_ID')
    DB_Properties_Filtered['OP_Accuracy'] = np.nan
    DB_Properties_Filtered['HS_Accuracy'] = np.nan
    
    Percentage_Difference_Total = []
    OAT_List = []
    LWT_List = []
    PLR_List = []
    PWR_List = []
    
    # Loop through each HP model
    for HP_Model in List_HP_Models:
        DSV = pd.read_excel('Heat_Pump_Data_F.xlsx', sheet_name=HP_Model)
        Capacity = RBFInterpolator(
            np.column_stack((DSV['OAT'], DSV['LWT'])), DSV['HC'], kernel=Interpolator, smoothing=0.0
        )
        Power = RBFInterpolator(
            np.column_stack((DSV['OAT'], DSV['LWT'])), DSV['IP'], kernel=Interpolator, smoothing=0.0
        )
    
        props_for_model = DB_Properties_Filtered.loc[
            DB_Properties_Filtered['HP_Model'] == HP_Model, 'Property_ID'
        ].tolist()
    
        # Loop through each property for this model
        for prop_id in props_for_model:
            counter += 1
            print(f"Progress: {counter}/{total_props} properties processed")
            if prop_id not in grouped.groups:
                continue
            df = grouped.get_group(prop_id).sort_values('Timestamp')
            diffs = []
            total_pred = total_act = 0.0
            # Loop through half-hourly points
            for i in range(1, len(df)):
                row, prev = df.iloc[i], df.iloc[i-1]
                if 35 < row['LWT'] < 55 and -10 < row['OAT'] < 10:
                    delta_Q  = float(row['Q'] - prev['Q'])*2
                    delta_PI = float(row['PI'] - prev['PI'])*2
                    if delta_PI == 0:
                        continue
                    hc_val = float(np.asarray(HC_FL(row['OAT'], row['LWT'])).item())
                    if hc_val == 0:
                        continue
                    plr = delta_Q / hc_val
                    if not (0.1 < plr <= 1):
                        continue
                    pred = float(np.asarray(PI(row['OAT'], row['LWT'], delta_Q, 'On_Off_Cd_0.9', 0.99)).item())
                    diffs.append((pred - delta_PI) / delta_PI * 100.0)
                    Percentage_Difference_Total.append((pred - delta_PI) / delta_PI * 100.0)
                    total_act += delta_PI
                    total_pred += pred
                    OAT_List.append(row['OAT'])
                    LWT_List.append(row['LWT'])
                    PLR_List.append(plr)
                    PWR_List.append(delta_PI)
            if diffs and total_pred > 0:
                DB_Properties_Filtered.loc[
                    DB_Properties_Filtered['Property_ID']==prop_id,
                    ['OP_Accuracy','HS_Accuracy']
                ] = float(np.mean(diffs)), float((total_pred-total_act)* 100.0 / total_act)
    
    # Remove known outlier properties
    DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH0241']
    DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH0584']
    DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH2307']

    # Clean difference values by removing -100 and out-of-range points
    cleaned_diffs = [
        x for x in Percentage_Difference_Total
        if not (
            (isinstance(x, np.ndarray) and x.size == 1 and x.item() == -100.0)
            or (isinstance(x, (float, np.float64)) and x == -100.0)
            or (isinstance(x, np.ndarray) and x.size == 1 and (x.item() > 100 or x.item() < -100))
            or (isinstance(x, (float, np.float64)) and (x > 100 or x < -100))
        )
    ]
    
    # Calculate boxplot statistics
    Q1, median, Q3 = np.percentile(cleaned_diffs, [25, 50, 75])
    IQR = Q3 - Q1
    lower_whisker = np.min([x for x in cleaned_diffs if x >= Q1 - 1.5 * IQR])
    upper_whisker = np.max([x for x in cleaned_diffs if x <= Q3 + 1.5 * IQR])
    mean = np.mean(cleaned_diffs)
    
    print("Boxplot values:")
    print(f"Min (after whisker): {lower_whisker}")
    print(f"Q1: {Q1}")
    print(f"Median: {median}")
    print(f"Q3: {Q3}")
    print(f"Max (after whisker): {upper_whisker}")
    print(f"Mean: {mean}")
    
    # Add results for this interpolation method
    update = {
        "Interpolator": Interpolator,
        "Average_Percentage_OP_Error": np.mean(cleaned_diffs),
        "Detached_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Detached']['OP_Accuracy'].mean(),
        "End-Terrace_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'End-Terrace']['OP_Accuracy'].mean(),
        "Semi-Detached_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Semi-Detached']['OP_Accuracy'].mean(),
        "Mid-Terrace_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Mid-Terrace']['OP_Accuracy'].mean(),
        "Detached_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Detached']['HS_Accuracy'].mean(),
        "End-Terrace_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'End-Terrace']['HS_Accuracy'].mean(),
        "Semi-Detached_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Semi-Detached']['HS_Accuracy'].mean(),
        "Mid-Terrace_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Mid-Terrace']['HS_Accuracy'].mean(),
        "5_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUHZ-W50VHA2']['OP_Accuracy'].mean(),
        "8_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM85VAA']['OP_Accuracy'].mean(),
        "11_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM112VAA']['OP_Accuracy'].mean(),
        "5_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUHZ-W50VHA2']['HS_Accuracy'].mean(),
        "8_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM85VAA']['HS_Accuracy'].mean(),
        "11_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM112VAA']['HS_Accuracy'].mean()
    }

    # Append results to summary DataFrame
    Comparison = pd.concat([Comparison, pd.DataFrame([update])], ignore_index=True)



#%% Regression

import pandas as pd
import numpy as np

# Define interpolation and performance functions
def HC_FL(OAT, LWT):
    return cap_coeffs[0]*OAT + cap_coeffs[1]*LWT + cap_coeffs[2]

def PI_FL(OAT, LWT):
    return pwr_coeffs[0]*OAT + pwr_coeffs[1]*LWT + pwr_coeffs[2]

def PI(OAT, LWT, HC, Op_Mode, Eff_SH):
    PLR = HC / HC_FL(OAT, LWT)
    COP_FL = HC_FL(OAT, LWT) / PI_FL(OAT, LWT)
    CF = np.interp(PLR, SPLCF['Part_Load'], SPLCF[Op_Mode])
    return (HC_FL(OAT, LWT) * PLR) / (CF * COP_FL)

SPLCF = pd.read_excel('Heat_Pump_Data_F.xlsx', sheet_name='Standard_PL_Correction_Factors')

# Define the column headings for the summary
columns = [
    "Interpolator",
    "Average_Percentage_OP_Error",
    "Detached_OP_Error",
    "End-Terrace_OP_Error",
    "Semi-Detached_OP_Error",
    "Mid-Terrace_OP_Error",
    "Detached_HS_Accuracy",
    "End-Terrace_HS_Accuracy",
    "Semi-Detached_HS_Accuracy",
    "Mid-Terrace_HS_Accuracy",
    "5_OP_Error",
    "8_OP_Error",
    "11_OP_Error",
    "5_HS_Accuracy",
    "8_HS_Accuracy",
    "11_HS_Accuracy",
]

# Container for results
Comparison = pd.DataFrame(columns=columns)

# Single interpolation approach used here
Interpolation_Method_List = ['regression']

# Load property metadata
DB_Properties = pd.read_csv(
    'Property_Data_Summary.csv',
    usecols=[
        'Property_ID', 'Excluded_SPF_analysis_reason', 'House_Form', 'HP_Model',
        'Cleansed_dataset_start', 'Cleansed_dataset_end'
    ]
)
DB_Properties['Cleansed_dataset_start'] = pd.to_datetime(
    DB_Properties['Cleansed_dataset_start'], errors='coerce', utc=True
)
DB_Properties['Cleansed_dataset_end'] = pd.to_datetime(
    DB_Properties['Cleansed_dataset_end'], errors='coerce', utc=True
)

# Models and heating periods
List_HP_Models = [
    'PUHZ-W50VHA2', 'PUZ-WM85VAA',
    'PUZ-WM112VAA'
]
heating_periods = [
    (pd.Timestamp('2020-10-01', tz='UTC'), pd.Timestamp('2021-04-01', tz='UTC')),
    (pd.Timestamp('2021-10-01', tz='UTC'), pd.Timestamp('2022-04-01', tz='UTC')),
    (pd.Timestamp('2022-10-01', tz='UTC'), pd.Timestamp('2023-04-01', tz='UTC'))
]

# Filter by coverage, data quality, models and house form
DB_Properties_Filtered = DB_Properties.loc[
    (DB_Properties['Cleansed_dataset_start'] <= heating_periods[-1][1]) &
    (DB_Properties['Cleansed_dataset_end'] >= heating_periods[0][0]) &
    (DB_Properties['Excluded_SPF_analysis_reason'] != 'Quality score outside threshold') &
    (DB_Properties['HP_Model'].isin(List_HP_Models)) &
    (DB_Properties['House_Form'].isin([
        'Detached','End-Terrace','Semi-Detached','Mid-Terrace'
    ]))
].reset_index(drop=True)

# Progress counters
total_props = DB_Properties_Filtered.shape[0]
counter = 0
print(f"Starting processing of {total_props} properties across multiple heating seasons...")

# Load half-hourly data in chunks and filter for selected properties and heating periods
all_props = DB_Properties_Filtered['Property_ID'].tolist()
chunks = pd.read_csv(
    'eoh_cleaned_half_hourly.csv',
    usecols=[
        'Timestamp','Property_ID','External_Air_Temperature',
        'Heat_Pump_Energy_Output','Whole_System_Energy_Consumed',
        'Heat_Pump_Heating_Flow_Temperature'
    ],
    parse_dates=['Timestamp'],
    chunksize=200_000
)
filtered = []
for chunk in chunks:
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], utc=True)
    # any of the defined heating periods
    mask = np.zeros(len(chunk), dtype=bool)
    for start, end in heating_periods:
        mask |= (chunk['Timestamp'] >= start) & (chunk['Timestamp'] <= end)
    mask &= chunk['Property_ID'].isin(all_props)
    filtered.append(chunk.loc[mask])

df_eoh = pd.concat(filtered, ignore_index=True)

# Clean and rename columns for modelling
df_eoh = (
    df_eoh.dropna(subset=[
        'External_Air_Temperature','Heat_Pump_Energy_Output',
        'Whole_System_Energy_Consumed','Heat_Pump_Heating_Flow_Temperature'
    ])
    .rename(columns={
        'External_Air_Temperature':'OAT',
        'Heat_Pump_Energy_Output':'Q',
        'Whole_System_Energy_Consumed':'PI',
        'Heat_Pump_Heating_Flow_Temperature':'LWT'
    })
)

# Group half-hourly data by property
grouped = df_eoh.groupby('Property_ID')
DB_Properties_Filtered['OP_Accuracy'] = np.nan
DB_Properties_Filtered['HS_Accuracy'] = np.nan

# Collections for analysis and diagnostics
Percentage_Difference_Total = []
OAT_List = []
LWT_List = []
PLR_List = []
PWR_List = []

for HP_Model in List_HP_Models:
    DSV = pd.read_excel('Heat_Pump_Data_F.xlsx', sheet_name=HP_Model)

    # Prepare regression inputs
    OAT_raw = DSV['OAT'].to_numpy()
    LWT_raw = DSV['LWT'].to_numpy()
    HC_raw  = DSV['HC'].to_numpy()
    IP_raw  = DSV['IP'].to_numpy()
    COP_raw = HC_raw / IP_raw

    # Linear regression design matrix [OAT, LWT, 1]
    A = np.column_stack((OAT_raw, LWT_raw, np.ones(len(DSV))))

    # Fit linear models for HC and IP
    cap_coeffs, _, _, _ = np.linalg.lstsq(A, HC_raw, rcond=None)
    pwr_coeffs, _, _, _ = np.linalg.lstsq(A, IP_raw, rcond=None)

    # Properties that use this model
    props_for_model = DB_Properties_Filtered.loc[
        DB_Properties_Filtered['HP_Model'] == HP_Model, 'Property_ID'
    ].tolist()

    for prop_id in props_for_model:
        counter += 1
        print(f"Progress: {counter}/{total_props} properties processed")
        if prop_id not in grouped.groups:
            continue
        df = grouped.get_group(prop_id).sort_values('Timestamp')
        diffs = []
        total_pred = total_act = 0.0

        # Iterate through half-hourly samples
        for i in range(1, len(df)):
            row, prev = df.iloc[i], df.iloc[i-1]
            if 35 < row['LWT'] < 55 and -10 < row['OAT'] < 10:
                delta_Q  = float(row['Q'] - prev['Q'])*2
                delta_PI = float(row['PI'] - prev['PI'])*2
                if delta_PI == 0:
                    continue
                # compute PLR from linear HC estimate and keep 0<PLR<=1 range
                hc_val = float(np.asarray(HC_FL(row['OAT'], row['LWT'])).item())
                if hc_val == 0:
                    continue
                plr = delta_Q / hc_val
                if not (0.1 < plr <= 1):
                    continue
                # predict input power from PI() and collect diagnostics
                pred = float(np.asarray(PI(row['OAT'], row['LWT'], delta_Q, 'On_Off_Cd_0.9', 0.99)).item())
                diffs.append((pred - delta_PI) / delta_PI * 100.0)
                Percentage_Difference_Total.append((pred - delta_PI) / delta_PI * 100.0)
                total_act += delta_PI
                total_pred += pred
                OAT_List.append(row['OAT'])
                LWT_List.append(row['LWT'])
                PLR_List.append(plr)
                PWR_List.append(delta_PI)

        # Store per-property accuracy metrics
        if diffs and total_pred > 0:
            DB_Properties_Filtered.loc[
                DB_Properties_Filtered['Property_ID']==prop_id,
                ['OP_Accuracy','HS_Accuracy']
            ] = float(np.mean(diffs)), float((total_pred-total_act)* 100.0 / total_act)

# Remove known outlier properties
DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH0241']
DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH0584']
DB_Properties_Filtered = DB_Properties_Filtered[DB_Properties_Filtered['Property_ID'] != 'EOH2307']

# Clean percentage differences: drop -100 and out-of-range values
cleaned_diffs = [
    x for x in Percentage_Difference_Total
    if not (
        (isinstance(x, np.ndarray) and x.size == 1 and x.item() == -100.0)
        or (isinstance(x, (float, np.float64)) and x == -100.0)
        or (isinstance(x, np.ndarray) and x.size == 1 and (x.item() > 100 or x.item() < -100))
        or (isinstance(x, (float, np.float64)) and (x > 100 or x < -100))
    )
]

    # Calculate boxplot stats
Q1, median, Q3 = np.percentile(cleaned_diffs, [25, 50, 75])
IQR = Q3 - Q1
lower_whisker = np.min([x for x in cleaned_diffs if x >= Q1 - 1.5 * IQR])
upper_whisker = np.max([x for x in cleaned_diffs if x <= Q3 + 1.5 * IQR])
mean = np.mean(cleaned_diffs)

print("Boxplot values:")
print(f"Min (after whisker): {lower_whisker}")
print(f"Q1: {Q1}")
print(f"Median: {median}")
print(f"Q3: {Q3}")
print(f"Max (after whisker): {upper_whisker}")
print(f"Mean: {mean}")

# Assemble summary record for this interpolation method
update = {
    "Interpolator": 'Regression',
    "Average_Percentage_OP_Error": np.mean(cleaned_diffs),
    "Detached_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Detached']['OP_Accuracy'].mean(),
    "End-Terrace_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'End-Terrace']['OP_Accuracy'].mean(),
    "Semi-Detached_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Semi-Detached']['OP_Accuracy'].mean(),
    "Mid-Terrace_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Mid-Terrace']['OP_Accuracy'].mean(),
    "Detached_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Detached']['HS_Accuracy'].mean(),
    "End-Terrace_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'End-Terrace']['HS_Accuracy'].mean(),
    "Semi-Detached_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Semi-Detached']['HS_Accuracy'].mean(),
    "Mid-Terrace_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['House_Form'] == 'Mid-Terrace']['HS_Accuracy'].mean(),
    "5_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUHZ-W50VHA2']['OP_Accuracy'].mean(),
    "8_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM85VAA']['OP_Accuracy'].mean(),
    "11_OP_Error": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM112VAA']['OP_Accuracy'].mean(),
    "5_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUHZ-W50VHA2']['HS_Accuracy'].mean(),
    "8_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM85VAA']['HS_Accuracy'].mean(),
    "11_HS_Accuracy": DB_Properties_Filtered[DB_Properties_Filtered['HP_Model'] == 'PUZ-WM112VAA']['HS_Accuracy'].mean()
}

# Append to the comparison table
Comparison = pd.concat([Comparison, pd.DataFrame([update])], ignore_index=True)
