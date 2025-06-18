import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Set global style using valid matplotlib style
plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Load data
data = pd.read_csv('realistic_ocean_climate_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])

## Data Preprocessing
# Temperature anomaly calculation (Δ°C from location baseline)
location_avg = data.groupby('Location')['SST (°C)'].mean().to_dict()
data['ΔSST (°C)'] = data.apply(lambda x: x['SST (°C)'] - location_avg[x['Location']], axis=1)

# Convert categorical variables
bleaching_map = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
data['Bleaching Severity'] = data['Bleaching Severity'].map(bleaching_map).fillna(0)
data = pd.get_dummies(data, columns=['Location'], drop_first=True)

## Model Training
X = data[['SST (°C)', 'ΔSST (°C)', 'pH Level', 'Bleaching Severity'] + 
       [col for col in data.columns if col.startswith('Location_')]]
y = data['Marine Heatwave'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'heatwave_model_v2.pkl')

## Visualization Functions
def create_temp_anomaly_plot(location):
    """Show temperature anomalies over time with heatwave events"""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    temp_data = data[data[f'Location_{location}'] == 1] if f'Location_{location}' in data.columns else None
    
    if temp_data is not None and not temp_data.empty:
        # Plot normal conditions
        normal = temp_data[~temp_data['Marine Heatwave']]
        ax.plot(normal['Date'], normal['ΔSST (°C)'], 
                'o-', color='deepskyblue', label='Normal Conditions')
        
        # Plot heatwave events
        heatwaves = temp_data[temp_data['Marine Heatwave']]
        ax.plot(heatwaves['Date'], heatwaves['ΔSST (°C)'], 
                'o-', color='crimson', label='Heatwave Events')
        
        # Annotate heatwave events
        for idx, row in heatwaves.iterrows():
            ax.annotate(f"Δ{row['ΔSST (°C)']:+.1f}°C", 
                       (row['Date'], row['ΔSST (°C)']), 
                       textcoords="offset points", xytext=(0,5), ha='center',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        plt.title(f'Temperature Anomalies in {location}\n(Δ°C from {location_avg[location]:.1f}°C baseline)', pad=20)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Temperature Anomaly (Δ°C)')
        plt.xlabel('Date')
        plt.legend()
    else:
        plt.title('Location data not available')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_combined_risk(probability, sst, delta_sst):
    """Radar chart showing risk factors"""
    categories = ['SST', 'ΔSST', 'Probability']
    values = [min(sst/35*100, 100), min((delta_sst+5)/10*100, 100), probability*100]  # Normalized and capped
    
    # Close the loop for radar chart
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, polar=True)
    
    ax.plot(angles, values, 'o-', linewidth=2, color='teal')
    ax.fill(angles, values, color='teal', alpha=0.25)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
    ax.set_rlabel_position(90)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 100)
    ax.set_title('Heatwave Risk Factors\n', pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()

## Prediction Function
def predict_with_enhanced_visuals():
    print("\n=== Marine Heatwave Prediction with Δ°C Analysis ===")
    
    # User inputs with numeric bleaching severity
    print("\nEnter the following details:")
    location = input("Location (e.g., 'Great Barrier Reef'): ")
    
    while True:
        try:
            sst = float(input("Current SST (°C): "))
            ph = float(input("pH Level: "))
            print("Bleaching Severity: 1=Low, 2=Medium, 3=High, 0=None")
            bleaching = int(input("Enter number (0-3): "))
            if bleaching not in [0,1,2,3]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input! Please enter numbers only.")
    
    # Calculate temperature anomaly
    delta_sst = sst - location_avg.get(location, 0)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'SST (°C)': [sst],
        'ΔSST (°C)': [delta_sst],
        'pH Level': [ph],
        'Bleaching Severity': [bleaching]
    })
    
    # One-hot encode location
    for loc in location_avg.keys():
        input_data[f'Location_{loc}'] = 1 if loc == location else 0
    
    # Ensure column alignment
    model = joblib.load('heatwave_model_v2.pkl')
    missing_cols = set(model.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[model.feature_names_in_]
    
    # Predict
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results
    print("\n=== Prediction Results ===")
    print(f"Probability: {probability*100:.1f}%")
    print(f"Prediction: {'HEATWAVE LIKELY' if probability > 0.5 else 'Normal Conditions'}")
    print(f"Temperature Anomaly: Δ{delta_sst:+.1f}°C from {location_avg.get(location, 'N/A'):.1f}°C baseline")
    
    # Show visualizations
    create_temp_anomaly_plot(location)
    plot_combined_risk(probability, sst, delta_sst)
    
    # Feature importance
    plt.figure(figsize=(10, 5), dpi=100)
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_
                         ).nlargest(10).sort_values()
    importance.plot(kind='barh', color='teal')
    plt.title('Top Predictive Factors for Marine Heatwaves', pad=20)
    plt.xlabel('Importance Score', labelpad=10)
    plt.ylabel('Features', labelpad=10)
    plt.tight_layout()
    plt.show()

# Run prediction
predict_with_enhanced_visuals()