#import pandas as pd
#import psycopg2
#from config import DATABASE_URL, LAST_HISTORICAL_DATE

#def get_db_connection():
    #"""Get database connection"""
    #return psycopg2.connect(DATABASE_URL)

#def load_all_data():
    #"""Load all data from database"""
    #conn = get_db_connection()
    #query = "SELECT * FROM bed_inventory"
    #df = pd.read_sql(query, conn)
    #conn.close()

    # Formatting the date time & filter til 31/12/2025
    #df["datetime"] = pd.to_datetime(df["datetime"])
    #df = df[df["datetime"].dt.date <= LAST_HISTORICAL_DATE]

    #return df


#def get_hospitals():
    #"""Get list of hospitals"""
    #conn = get_db_connection()
    #query = """
    #SELECT DISTINCT hospital 
    #FROM bed_inventory 
    #ORDER BY hospital
    #"""

    #df = pd.read_sql(query, conn)
    #conn.close()

    #return df["hospital"].tolist()


#def get_hospital_wards(hospital_name):
    #"""Get wards of a hospital"""
    #conn = get_db_connection()
    #query = """
    #SELECT DISTINCT ward 
    #FROM bed_inventory 
    #WHERE hospital = %s 
    #ORDER BY ward
    #"""
    #df = pd.read_sql(query, conn, params=[hospital_name])
    #conn.close()

    #return df["ward"].tolist()



import pandas as pd
import os
from config import LAST_HISTORICAL_DATE

# Path to CSV file
DATA_PATH = "data/bed_inventory.csv"


def load_all_data():
    """
    Load all bed inventory data from CSV file
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # Ensure datetime format
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter historical data
    df = df[df["datetime"].dt.date <= LAST_HISTORICAL_DATE]

    return df


def get_hospitals():
    """Get list of hospitals"""
    df = load_all_data()

    hospitals = (
        df["hospital"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return hospitals


def get_hospital_wards(hospital_name):
    """Get wards for a selected hospital"""
    df = load_all_data()

    wards = (
        df[df["hospital"] == hospital_name]["ward"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return wards
