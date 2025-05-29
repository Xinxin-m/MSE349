# Crypto Data Analysis Project

This repository contains code and data for analyzing cryptocurrency market data from Kraken exchange. The project processes OHLCVT (Open, High, Low, Close, Volume, Trades) data and generates various derived datasets and characteristics.

## Project Structure

### Data Folder Structures

- `Kraken_OHLCVT/`: Contains raw OHLCVT data from Kraken exchange
  - Data is stored in compressed format (Kraken_OHLCVT_Q1_2024.zip)
  - Contains detailed market data for various cryptocurrency pairs

- `USD_60/`: Contains processed parquet files with 60-minute interval data
  - Generated from `process_data.py`
  - Contains cleaned and processed market data
  - Files are stored in parquet format for efficient storage and retrieval

- `USD_60_2022-2025/`: Contains processed data for the specific time period
  - Subset of the USD_60 data focused on 2022-2025 timeframe

- `USD_60_indicators/`: Contains derived indicators and characteristics
  - Generated from `process_data2.py`
  - Contains additional market characteristics and indicators

### Key Files
- `kraken.ipynb`: Analysis of initial Kraken OHLCVT data and summary generation
- `process_data.py`: Processes raw data into USD_60 parquet files
- `characteristics.ipynb`: Development of market characteristics (converted to process_data2.py)
- `process_data2.py`: Adds characteristics to the USD_60 files
- `plot_existing_entries.py`: Visualization of data entries
- `pca.ipynb`: Principal Component Analysis of the processed data (to be completed)




1. **Data Collection & Visualization**
   - Raw data downloaded fron (Kraken)[https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data] is stored in `Kraken_OHLCVT/`, it contains spot trading pairs OHLCVT data upto 2025-Q1 (2025-03-31)
   - Initial analysis and summary CSVs are generated using `kraken.ipynb`, these files list the first and last timestamps and dates, and the number of rows contained in each dataset
   - MarketCap data for the entire crypto market (top 100 coins) is retrieved from Coindesk using the free API
   - `get_data.ipynb` contains the code for API calls
   - `plot_existing_entries.py` plots existing and missing entries for all the datasets listed

2. **Data Processing**
   - `process_data.py` converts csv data into parquet files, adding a few new columns (such as `return` and `lor_return`) without filling missing rows
   - Processed data is stored in `USD_60/` directory
   - Data is cleaned and standardized during this process

3. **Characteristic Generation**
   - `process_data2.py` fills missing rows with the correct timestamps and `NaN` elsewhere, and adds ~30 market characteristics
   - Results are stored in `USD_60_indicators/`

### Data Files

- Market cap data files (ignored in git):
  - `marketcap_data.csv`: the raw marketcap data downloaded via CoinDesk API
  - `mcap_processed.csv`: processed marketcap data (using `kraken.ipynb`) that now contains `return` and `log_return`)

- Summary and metadata files:
  - We consider only coins paired with USD on a 60-minute interval (therefore the `USD_60` prefix):
    - `USD_60_summary.csv`
    - `USD_60_filenames_parquet.txt`
    - `USD_60_filenames_csv.txt`



### Data Processing Workflow v.2: PCA

1. Build characteristics for the USD 720 datasets:
   - `process_data` and `process_data_2_720` and `compute_betas` (adding only marketcap beta), save resulting `.parquet` datasets to the folder `USD_720`
2. For PCR with the characteristic matrix of dimension `NT\times P` , we need to concatenate all the datasets within the desired timeframe into one dataframe first, this is done via

   - `prepare_dataset.ipynb`: get list of dataset names that are 'valid' (i.e. meet missing entry threshold) for a timeframe of your choice (here using monthly windows of 60 timesteps since `720=12h`), save the list of filenames
   - `timestamp_coverage.py`: concatenate into one dataframe
   - `model_analysis.py` and/or `pca_analysis.py` split the dataset into train/validation/test, perform prediction, analyze and plot MSE

The above scripts can be customized.


### Note

Some data files are intentionally excluded from version control (see `.gitignore`):
- Raw data files (Kraken_OHLCVT)
- Large processed data files
- Market cap data files
- Environment files (.env) 

### Note on Data Resources

- One can get daily or hourly volume from different exchanges on CoinDesk [https://developers.coindesk.com/documentation/legacy/Historical/dataExchangeSymbolHistoday]
