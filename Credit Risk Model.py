import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    # Load the credit card data file
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    #Prepare features and target for modeling
    # Drop 'Time' column
    X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    y = df['Class'].astype(int)
    print(f"Fraud cases: {sum(y)}, Legit cases: {len(y) - sum(y)}")
    print(f"Fraud rate: {sum(y)/len(y):.5f}")
    
    # Handle missing values if any
    X = X.fillna(X.median())
    return X, y

def calculate_risks(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Predict probabilities on full dataset (for risk distribution)
    proba = model.predict_proba(X)[:, 1]
    risk_perc = np.clip(proba, 0.0001, 0.9999) * 100
    return risk_perc

def create_visualization(df):
    # Create risk distribution plot
    plt.figure(figsize=(14, 6))
    ax = sns.histplot(df['Default_Risk_Perc'], bins=50, kde=True, color='royalblue', alpha=0.7)
    median_risk = df['Default_Risk_Perc'].median()
    q75_risk = df['Default_Risk_Perc'].quantile(0.75)
    ax.axvline(median_risk, color='darkorange', linestyle='--', linewidth=2)
    ax.axvline(q75_risk, color='crimson', linestyle='--', linewidth=2)
    ax.text(median_risk + 0.05, ax.get_ylim()[1]*0.9, f'Median: {median_risk:.2f}%', color='darkorange')
    ax.text(q75_risk + 0.05, ax.get_ylim()[1]*0.8, f'75th percentile: {q75_risk:.2f}%', color='crimson')
    plt.title('Distribution of Predicted Fraud Risk Percentages')
    plt.xlabel('Predicted Fraud Risk (%)')
    plt.ylabel('Number of Transactions')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    print("Starting Credit Card Fraud Risk Analysis\n" + "="*40)
    df = load_data('/Users/johnny/Desktop/Projects/Credit Risk Project/creditcard.csv')
    X, y = preprocess_data(df)
    risk_perc = calculate_risks(X, y)
    df['Default_Risk_Perc'] = risk_perc
    df[['Default_Risk_Perc']].to_csv('/Users/johnny/Desktop/Projects/Credit Risk Project/creditcard_risk_percentages.csv', index=False)
    print("Risk percentages saved to: creditcard_risk_percentages.csv")
    create_visualization(df)
    print(f"\nFound {len(df)} transactions")
    print(f"Average predicted risk: {df['Default_Risk_Perc'].mean():.2f}%")
    print(f"Risk range: {df['Default_Risk_Perc'].min():.2f}% - {df['Default_Risk_Perc'].max():.2f}%")
    print(f"Median risk: {df['Default_Risk_Perc'].median():.2f}%")
    print(f"75th percentile risk: {df['Default_Risk_Perc'].quantile(0.75):.2f}%")

if __name__ == "__main__":
    main()
