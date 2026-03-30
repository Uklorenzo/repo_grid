This contains grid frequency data.
This script loads raw grid frequency data for a specified year and month, performs cleaning to remove outliers and artifacts, and generates various plots to analyze the frequency behavior. The cleaned data is saved in a compressed format for future use.
The script is designed to handle common file naming conventions and structures, but may require adjustments for specific datasets. It includes detailed comments and explanations for each step of the process.
Note: Adjust the 'location', 'year', and 'month' variables at the top of the script to point to your data files. The script will attempt to find the correct file based on common naming patterns.
Data Source: European grid frequency data, typically recorded at 1-second intervals. The script assumes the frequency is provided in mHz relative to 50 Hz (e.g., 20 mHz means 50.020 Hz). If the data appears to be in absolute Hz, it will be converted accordingly.
The script performs the following steps:
1. Load the raw data from a CSV file, handling different possible schemas.
2. Normalize the time series to have a consistent index with 1-second intervals, filling missing timestamps with NaN.
3. Identify and remove outliers based on absolute frequency thresholds and large jumps.
4. Generate a quality plot to visualize the raw frequency data and highlight identified issues.
5. Plot the cleaned frequency time series and histogram.


