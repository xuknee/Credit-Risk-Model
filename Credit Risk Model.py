import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set style for better looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

def load_data(file_path):
    """Load and validate the data file"""
    try:
        df = pd.read_csv(file_path, sep=';')
        print("Data loaded successfully.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")

def preprocess_data(df):
    """Clean and prepare the data for modeling"""
    # Check and convert Creditability
    if 'Creditability' not in df.columns:
        raise ValueError("Missing 'Creditability' column")
    
    # Convert to numeric (1 for Bad, 0 for Good)
    df['Creditability'] = df['Creditability'].replace({
        'Good': 0,
        'Bad': 1,
        '0': 0,
        '1': 1
    }).astype(int)
    
    # Check if we have both classes
    class_counts = df['Creditability'].value_counts()
    print("Creditability distribution:")
    
    if len(class_counts) < 2:
        print("Warning: Only one credit class found in data")
        # Add dummy data if missing one class (for demonstration)
        if 0 not in class_counts:
            dummy = df.iloc[0].copy()
            dummy['Creditability'] = 0
            df = df.append(dummy, ignore_index=True)
        if 1 not in class_counts:
            dummy = df.iloc[0].copy()
            dummy['Creditability'] = 1
            df = df.append(dummy, ignore_index=True)
        print("Added temporary dummy records to enable modeling")
    
    # Convert other categorical variables
    cat_mappings = {
        'Account_Balance': {
            'No account': 3, 
            'None (No balance)': 2, 
            'Some Balance': 1
        },
        'Value_Savings_Stocks': {
            'None': 3, 
            'Below 100 DM': 2, 
            '[100, 1000] DM': 1, 
            'Above 1000 DM': 0
        }
    }
    
    for col, mapping in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(2)  # Default to medium risk
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
        df[col] = np.clip(df[col], 
                         df[col].quantile(0.01),
                         df[col].quantile(0.99))
    
    return df

def calculate_risks(df):
    """Calculate risk percentages for all applicants"""
    X = df.drop('Creditability', axis=1)
    y = df['Creditability']
    
    # Create simple model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    # Train on all data
    model.fit(X, y)
    
    # Get probabilities
    proba = model.predict_proba(X)[:, 1]
    df['Default_Risk_Perc'] = np.clip(proba, 0.01, 0.99) * 100
    return df

def create_visualization(df):
    """Create risk distribution plot using Seaborn/Matplotlib"""
    plt.figure(figsize=(14, 6))
    
    # Histogram of risk percentages
    ax = sns.histplot(data=df, x='Default_Risk_Perc', bins=20, 
                     kde=True, color='royalblue', alpha=0.7)
    
    # Add vertical lines for important percentiles
    median_risk = df['Default_Risk_Perc'].median()
    q75_risk = df['Default_Risk_Perc'].quantile(0.75)
    
    ax.axvline(median_risk, color='darkorange', linestyle='--', linewidth=2)
    ax.axvline(q75_risk, color='crimson', linestyle='--', linewidth=2)
    
    # Add annotations
    ax.text(median_risk+1, ax.get_ylim()[1]*0.9, 
           f'Median: {median_risk:.1f}%', color='darkorange')
    ax.text(q75_risk+1, ax.get_ylim()[1]*0.8, 
           f'75th %ile: {q75_risk:.1f}%', color='crimson')
    
    # Formatting
    plt.title('Distribution of Default Risk Percentages', fontsize=16, pad=20)
    plt.xlabel('Default Risk Percentage', fontsize=12)
    plt.ylabel('Number of Applicants', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.show()

def main():
    print("Starting Credit Risk Analysis\n" + "="*30)
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        df = load_data('german_credit_data.csv')
        
        # 2. Preprocess data
        print("2. Cleaning and preparing data...")
        df = preprocess_data(df)
        
        # 3. Calculate risks
        print("3. Calculating risk percentages...")
        df = calculate_risks(df)
        
        # 4. Save results
        print("4. Saving results...")
        df[['Default_Risk_Perc']].to_csv('risk_percentages.csv', index=False)
        print("Risk percentages saved to: risk_percentages.csv")
        
        # 5. Create visualization
        print("5. Creating visualization...")
        create_visualization(df)
        
        print("\nAnalysis completed successfully!")
        print(f"Found {len(df)} applicants")
        print(f"Average risk: {df['Default_Risk_Perc'].mean():.1f}%")
        print(f"Risk range: {df['Default_Risk_Perc'].min():.1f}%-{df['Default_Risk_Perc'].max():.1f}%")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("- Verify 'Creditability' column contains 'Good'/'Bad' or 0/1")
        print("- Check for missing values in critical columns")
        print("- Ensure file uses semicolons (;) as separators")
        print("- The first row should contain column headers")

if __name__ == "__main__":
    main()