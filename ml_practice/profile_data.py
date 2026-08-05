import pandas as pd
from ydata_profiling import ProfileReport

def main():
    print("Loading data.csv...")
    try:
        df = pd.read_csv('data.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Failed to load data.csv: {e}")
        return

    print("Generating pandas profiling report...")
    try:
        profile = ProfileReport(df, title="Pandas Profiling Report for data.csv", explorative=True)
        
        output_path = 'data_profile_report.html'
        print(f"Saving report to {output_path}...")
        profile.to_file(output_path)
        print("Report saved successfully!")
    except Exception as e:
        print(f"Failed to generate or save report: {e}")

if __name__ == "__main__":
    main()
