import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.download_files import GetFiles
import json
sys.path.insert(0, str(Path(__file__).parent.parent))



class EDA():
    """
    Exploratory Data Analysis class for cybersecurity attack"""

    def __init__(self):
        """
        Initializes the EDA class loading the necessary 
        variables in memory and setting up the necessary
        directories and downloading the necessary files
        
        Returns nothing
        """
        self.steps_to_skip = dict()
        self.run_full_analysis : bool
        self.runstep : bool
        self.step_short: str
        # Defines the directory where to store the datasets
        self.data_dir = str(Path("data")) + "/"
        # Define the components of the filename of the dataset for each step of the EDA
        self.dsbasename = "cyberds"
        self.step = str()
        self.ext = ".parquet"
        # Defines the file to be downloaded ad tested for presence 
        # at the beginning of the initialization of EDA class.
        self.required_files = [
            self.data_dir + 'alternateNamesV2.txt',
            self.data_dir + 'IN.txt',
            self.data_dir + 'admin1CodesASCII.txt'
        ]
        
        # Add file names and file urls to the dict below to download and check for after download 
        files_to_download = {
            'alternateNamesV2.zip': 'https://download.geonames.org/export/dump/alternateNamesV2.zip',
            'IN.zip': 'https://download.geonames.org/export/dump/IN.zip',
            'admin1CodesASCII.txt': 'https://download.geonames.org/export/dump/admin1CodesASCII.txt',
            'cybersecurity_attacks.csv' : "https://learn.dsti.institute/pluginfile.php/45207/mod_assign/introattachment/0/Project%201.zip?forcedownload=1"
        }
            
        # Mumbai/Delhi are ~20-30 million - anything above is likely a region/country entry
        # Find the most populated actual city (usually Mumbai or Delhi ~20-30M)
        self.max_city_population = 30_000_000  # 30 million as safe threshold
        
        # Download the necessary files
        self.download_files(required_files=self.required_files)
        
        # Load main India cities
        self.india_df = pd.read_csv( str(self.data_dir) + 'IN.txt', sep='\t', header=None,
            usecols=[0, 1, 4, 5, 10, 14],
            names=['geonameid', 'name', 'lat', 'lon', 'admin1_code', 'population']
        )
        
        self.admin_df = pd.read_csv(str(self.data_dir) + 'admin1CodesASCII.txt', sep='\t', header=None,
            names=['code', 'state', 'state_ascii', 'geonameid'])
        
        # Convert to string and handle NaN
        self.india_df['admin1_code'] = self.india_df['admin1_code'].fillna('').astype(str).str.replace('.0', '', regex=False)
        
        
        # Original dataset import to df
        self.cybersecurity_df = pd.read_csv(str(self.data_dir) + "cybersecurity_attacks.csv" )
        
        self.step_short = self.get_first_step()
        
    def download_files(self, required_files):
        """
        Check for required files and download them if they are missing.

        This method verifies that all required files exist in the data directory.
        If any files are missing, it initiates the download process using the
        GetFiles class.

        Args:
            required_files (list): List of file paths to check for existence.

        Returns:
            None

        Side Effects:
            - Downloads missing files to self.data_dir
            - Prints status message if all files are already present
        """
        if not all(Path(f).exists() for f in required_files):
            GetFiles(required_files, Path(self.data_dir))
        else:
            print("All files already present, skipping download.")
        
    def find_city_coords(self, cities: dict, city: str, state: str) -> list[str] | None:
        """
        Find coordinates for a city by trying different city-state combinations.

        Attempts to find geographic coordinates by checking both (city, state)
        and (state, city) combinations in the cities dictionary. This handles
        cases where city and state might be reversed in the input data.

        Args:
            cities (dict): Dictionary mapping (city, state) tuples to coordinate data.
            city (str): Name of the city to search for.
            state (str): Name of the state to search for.

        Returns:
            list[str] | None: List containing coordinate data if found, None otherwise.

        Example:
            coords = self.find_city_coords(cities_dict, "Mumbai", "Maharashtra")
        """

        if cities.get((city, state)) is not None:
            coords = cities.get((city, state))
        elif cities.get((state, city)) is not None:
            coords = cities.get((state, city))
        else:
            return None
        return coords

    def city_state_to_coords(self, cities: dict, cities_states: str) -> pd.Series:
        """
        Convert a city-state string to latitude and longitude coordinates.

        Parses a comma-separated city-state string, looks up the city in the
        provided dictionary, and returns coordinates as a pandas Series.

        Args:
            cities (dict): Dictionary mapping city names to coordinate dictionaries
                containing 'lat' and 'lon' keys.
            cities_states (str): Comma-separated string in format "City, State".

        Returns:
            pd.Series: Series containing [latitude, longitude]. Returns [None, None]
                if the city is not found or if an error occurs during parsing.

        Example:
            coords = self.city_state_to_coords(cities_dict, "Mumbai, Maharashtra")
            # Returns pd.Series([19.0760, 72.8777])
        """
        try:
            city = cities_states.split(', ')[0]
            coords = cities.get(city)
            if coords:
                return pd.Series([coords['lat'], coords['lon']])
            return pd.Series([None, None])
        except Exception as e:
            print(f"error {e}")
            return pd.Series([None, None])
        
    def clean_geolocation_column(self) -> None:
        """

        This method performs:

        1. Merges Indian administrative regions with city data
        2. Analyzes and cleans population data
        3. Removes outliers and duplicates based on population thresholds
        4. Creates city-based lookup dictionaries for geocoding
        5. Attempts geocoding using multiple strategies:
           - Exact city-state matching
           - City-only matching (highest population)
           - Alternate names matching
           - Historical names matching
        6. Exports processed geographic data and missing data to parquet files

        Args:
            None

        Returns:
            None

        Side Effects:
            - Modifies self.india_df and self.admin_df
            - Creates parquet files: 'india_cities.parquet', 'geo_data.parquet',
              'missing_data.parquet'
            - Prints extensive analysis output including statistics, missing values,
              and processing time

        Performance:
            Prints total execution time at completion.
        """
        self.admin_df = self.admin_df[self.admin_df['code'].str.startswith('IN.')]
        self.admin_df['admin1_code'] = self.admin_df['code'].str.split('.').str[1]

        self.india_df = self.india_df.merge(self.admin_df[['admin1_code', 'state']], on='admin1_code', how='left')

        # --- Analysis before dropping ---
        has_state = self.india_df['state'].notna()

        print("=== Population comparison: Missing vs Present state ===\n")

        print(f"Rows WITH state: {has_state.sum()}")
        print(f"Rows WITHOUT state: {(~has_state).sum()}\n")

        print("Population stats - WITH state:")
        print(self.india_df.loc[has_state, 'population'].describe())

        print("\nPopulation stats - WITHOUT state:")
        print(self.india_df.loc[~has_state, 'population'].describe())

        print("\n=== Top 20 cities WITHOUT state (by population) ===")
        missing_state = self.india_df[~has_state].nlargest(20, 'population')[['name', 'population']]
        print(missing_state.to_string(index=False))

        print("\n=== Top 20 cities WITH state (by population) ===")
        with_state = self.india_df[has_state].nlargest(20, 'population')[['name', 'state', 'population']]
        print(with_state.to_string(index=False))

        # Drop rows where population is 0 AND state is missing
        condition = (self.india_df['population'] == 0) & (self.india_df['state'].isna())
        print(f"Dropping {condition.sum()} rows with population=0 and missing state")

        self.india_df = self.india_df[~condition]

        # Drop rows where population is 0
        condition = (self.india_df['population'] == 0)
        print(f"Dropping {condition.sum()} rows with population=0")

        self.india_df = self.india_df[~condition]

        # Check the highest population values first
        print("=== Top 10 entries by population ===")
        print(self.india_df.nlargest(10, 'population')[['name', 'state', 'population']].to_string(index=False))

        outliers = self.india_df[self.india_df['population'] > self.max_city_population]
        print(f"\n=== Rows with population > {self.max_city_population:,} ===")
        print(outliers[['name', 'state', 'population']].to_string(index=False))

        print(f"\nDropping {len(outliers)} rows")
        self.india_df = self.india_df[self.india_df['population'] <= self.max_city_population]

        # Drop duplicates, keeping the one with highest population
        self.india_df = self.india_df.sort_values('population', ascending=False)
        self.india_df = self.india_df.drop_duplicates(subset=['name', 'state'], keep='first')

        self.india_df.to_parquet(str(self.data_dir) + 'india_cities.parquet')

        column_names = ["Geolocation Lat", "Geolocation Long"]

        df_geo_data = pd.DataFrame(self.cybersecurity_df["Geo-location Data"])
        for i in range(len(column_names)):
            df_geo_data.insert( i , column_names[i] , value = np.nan )
            print(f"Inserting column {column_names[i]}")
        print(df_geo_data.head())

        # Check how many are missing
        total = len(df_geo_data)
        missing = df_geo_data["Geolocation Lat"].isna().sum()
        print(f"Missing: {missing}/{total} ({100*missing/total:.1f}%)")

        # See which cities aren't being found
        df_geo_data["city_state"] = self.cybersecurity_df["Geo-location Data"]
        not_found = df_geo_data[df_geo_data["Geolocation Lat"].isna()]["city_state"].value_counts()
        print("\n=== Top 30 cities not found ===")
        print(not_found.head(30))

        # Rebuild lookup with just city names (keep highest population for duplicates)
        city_only = self.india_df.sort_values('population', ascending=False).drop_duplicates(subset=['name'], keep='first')
        cities_by_name = city_only.set_index('name')[['lat', 'lon', 'state', 'population']].to_dict('index')
        # Re-run geocoding
        df_geo_data[["Geolocation Lat", "Geolocation Long"]] = self.cybersecurity_df["Geo-location Data"].apply(
            lambda x: self.city_state_to_coords(cities_by_name, x)
        )

        # Check again
        missing = df_geo_data["Geolocation Lat"].isna().sum()
        print(f"Missing after city-only matching: {missing}/{len(df_geo_data)} ({100*missing/len(df_geo_data):.1f}%)")

        # Check what's still not found
        not_found = df_geo_data[df_geo_data["Geolocation Lat"].isna()]["city_state"].apply(
            lambda x: x.split(', ')[0]
        ).value_counts()

        print(f"Unique cities not found: {len(not_found)}")
        print("\n=== Top 30 cities still missing ===")
        print(not_found.head(30))

        # Load alternate names (this file is large, filter for India geonameids)
        india_ids = set(self.india_df['geonameid'])

        # Carica alternate names con flag storico
        alt_names = pd.read_csv(self.data_dir + '/alternateNamesV2.txt', sep='\t', header=None,
            usecols=[1, 3, 7],
            names=['geonameid', 'alt_name', 'is_historic'],
            dtype={'is_historic': str}
        )

        # Filtra per India
        alt_names = alt_names[alt_names['geonameid'].isin(india_ids)]

        # Crea lookup che include ANCHE i nomi storici
        alt_names_all = alt_names.merge(
            self.india_df[['geonameid', 'lat', 'lon', 'population']], 
            on='geonameid'
        )

        # Aggiungi al lookup combinato
        alt_lookup = (alt_names_all
            .sort_values('population', ascending=False)
            .drop_duplicates(subset='alt_name', keep='first')
            .set_index('alt_name')[['lat', 'lon']]
            .to_dict('index'))

        cities_combined = {**alt_lookup, **cities_by_name}

        print(f"Total Lookup (with historical names): {len(cities_combined)}")

        df_geo_data[["Geolocation Lat", "Geolocation Long"]] = self.cybersecurity_df["Geo-location Data"].apply(
            lambda x: self.city_state_to_coords(cities_combined, x)
        )


        alt_names = pd.read_csv(self.data_dir + 'alternateNamesV2.txt', sep='\t', header=None,
            usecols=[1, 3, 7],  # geonameid, alt_name, isHistoric
            names=['geonameid', 'alt_name', 'is_historic']
        )
        
        # Include both current and historic names
        historic_names = alt_names[alt_names['is_historic'] == 1]
        print(f"Hystoric name found: {len(historic_names)}")
        missing = df_geo_data["Geolocation Lat"].isna().sum()
        print(f"Missing: {missing}/{len(df_geo_data)} ({100*missing/len(df_geo_data):.1f}%)")

        # Create boolean mask: True where Lat is missing (NaN)
        missing_mask = df_geo_data["Geolocation Lat"].isna()

        df_geo_data.to_parquet(str(self.data_dir) + 'geo_data.parquet')

        missing_data_df = df_geo_data.loc[missing_mask]


        print(missing_data_df.head())
        missing_data_df.to_parquet(str(self.data_dir) + 'missing_data.parquet')
        
        
        

    def split_datetime_column(self) -> None:
        """
        This function splits the Timestamp column in to two columns for date and time.
        Can optionally drop the column if drop original column is set to True (to be implemented)
        
        """
        
        self.cybersecurity_df.Timestamp = self.cybersecurity_df.Timestamp.apply(pd.to_datetime)
        
        print(self.cybersecurity_df)
        self.cybersecurity_df['Day'] = [d.date() for d in self.cybersecurity_df['Timestamp']]
        self.cybersecurity_df['Time'] = [d.time() for d in self.cybersecurity_df['Timestamp']]
        print(self.cybersecurity_df)
        
        
        
    def update_filename(self):
        """
        Updates the filename of the dataset parquet file each time is called.
        Ideally should be called after each (major) step of EDA or cleaning        
        """
        filename = self.dsbasename + self.step_short + self.ext
        self.dsfile_relative_path = self.data_dir + filename

    def load_file_from_step(self) -> None:
        """
        Checks the existence and loads a file for the given step of the analysis        
        """
        if self.run_full_analysis:
            pass
        else:
            try:
                f = self.data_dir + self.dsbasename + self.step_short + self.ext
                Path(f).exists()
                self.cybersecurity_df = pd.read_parquet(f)
            except FileExistsError:
                print(f"file {f} does not exist")
                pass
            
    def print_step_artwork(self, beginning=True):
        
        """
        Prints the long step name and an artwork for beginning and end of step according to the beginning flag state
        """
        if beginning:
            print("="*60+"\n\n")
            
            print(f"Beginning of {self.step_long}\n")
            
            print("="*60+"\n\n")
        else:
            print("="*60+"\n\n")
            
            print(f"End of {self.step_long}\n")
            
            print("="*60+"\n\n")
        
    def set_skip_steps(self, analysis_steps: dict):
        """
        Sets values of internal dict self.steps_to_skip according to steps_to_skips values
        Intended to be used to skip certain parts of the analysis by setting values of the passed dict to True
        """
        
        for k,v in self.steps_to_skip:
            for key, val in analysis_steps:
                if k == key and v != val:
                    v = val
                else:
                    pass
                
    def get_skip_step(self, current_step: str) -> bool:
        
        """
        Gets current_step as input and tests if the step needs to be skipped. If the
        step is not in self.steps_to_skips dict calls add.step with current step as parameter
        to add it to steps to be skipped for the next execution.
        Returns:
            bool : True if the step has already run or if it set to be skipped False otherwise
        """
        if current_step in self.steps_to_skip.keys():
            return self.steps_to_skip[current_step]
        else:
            self.add_step(current_step)
            return False
    
    def add_step(self, step: str):
        """
        Adds step to self.steps_to_skip if not already present in it

        Args:
            step (str): name of the step of the analysis passed as argument
        """
        if step in self.steps_to_skip.keys():
            pass
        else:
            self.steps_to_skip[step]= True
    
    def get_next_step(self):
        """
        Gets the next step of the analysis to be run based on the self.steps_to_skip dict
        """
        print(f"current step is: {self.step_short}")
        try:
            for k, v in self.steps_to_skip.items():
                if v:
                    print(f"skipped step is {k}")
                    pass
                else:
                    print(f"next running step is {k}")
                    return k
        except KeyError as e:
            print(e)
            exit(1)
            
                
    def get_first_step(self) -> str:
        """
        Gets the first step of the analysis set in the settings json

        Returns:
            str: the first step of the analysis as a string
        """
        with open("settings/settings.json") as f:
            settings = json.loads(f.read())
        print(settings)
        self.steps_to_skip = settings["steps_to_skip"]
        if not any(self.steps_to_skip.values()):
            self.run_full_analysis = True
            return "_step1"
        else:
            for k, v in self.steps_to_skip.items():
                if not v:
                    step_to_run = k
                    break
                else:
                    continue
            return step_to_run
            
                        
    def run_EDA(self):
        
        start_t = time.time()
        
        match self.step_short:
            case "_step1":
                if not self.get_skip_step(self.step_short):
                    #Step 1
                    self.step_long = "Step #1 cleaning of the geolocation column"
                    self.print_step_artwork()

                    self.clean_geolocation_column()
                    
                    self.update_filename()
                    
                    print(f"Saving file {self.dsfile_relative_path}")

                    self.cybersecurity_df.to_parquet( self.dsfile_relative_path)
                    
                    self.print_step_artwork(beginning=False)
                    self.steps_to_skip[self.step_short] = True
                    self.step_short = self.get_next_step()
                    self.run_EDA()
                else:                    
                    self.steps_to_skip[self.step_short] = True
                    self.load_file_from_step()
                    self.step_short = self.get_next_step()
                    self.run_EDA()    
                    
            case "_step2": 
                if not self.get_skip_step(self.step_short):
                    self.step_long = "Step #2 split Timestamp column"
                    self.print_step_artwork()
                    #Step2
                    self.split_datetime_column()
                    
                    self.update_filename()
                    
                    print(f"Saving file {self.dsfile_relative_path}")

                    self.cybersecurity_df.to_parquet( self.dsfile_relative_path)
                    
                    self.print_step_artwork(beginning=False)
                    self.steps_to_skip[self.step_short] = True
                    self.step_short = self.get_next_step()
                    self.run_EDA()
                else:
                    self.steps_to_skip[self.step_short] = True
                    self.load_file_from_step()
                    self.step_short = self.get_next_step()
                    self.run_EDA()
        
            case "_step3":
                self.step_long = "Step #2 split Timestamp column"
                self.print_step_artwork()
                #Step3
                
                    
        
        
        tot_t = time.time()

        
        print(f"total run time = {tot_t - start_t}")