"""
Module for geo located tweets data collection and home location inference.
Twitter/SoSweet/snapshot data.
"""

import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import geopandas as gpd
import json
import os
import tarfile
from tqdm import tqdm
import osmnx as ox

DATA_PATH = "../data/raw/snapshot_mount/"
SAVE_DATA_PATH = "../data/interim/"
LON_LAT_PROJ = "epsg:4326"
EUROPEAN_PROJ = "epsg:3035"

def collect_geo_located_tweets(snapshot_data_path: str)->pl.DataFrame:
    """Collect all geo located data. Note: snapshot and rtw data don't have the same format.
    
    Args:
        snapshot_data_path: main folder containing .tgz files.
    Returns:
        polars DataFrame collecting all tweets with available geo information.
    """

    schema = {
        "tweetId": pl.Utf8,
        "body": pl.Utf8,
        "postedTime": pl.Utf8,
        "userId": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
    }
    geo_tweets = []

    # Iterate over all tgz files
    all_tgz_files = os.listdir(snapshot_data_path)
    print(f"{len(all_tgz_files)} folders to be processed.")

    for tgz_file in tqdm(all_tgz_files, total= len(all_tgz_files), desc=f"Processing tgz files in {snapshot_data_path}"):
        if tgz_file.endswith(".tgz") and tgz_file!="snapshot.tgz":
            tgz_path = os.path.join(snapshot_data_path, tgz_file)
            
            with tarfile.open(tgz_path, "r:gz") as tar:
                # Iterate through each file inside the archive
                for member in tqdm(tar.getmembers(), desc=f"Processing {tgz_file}", disable=True):
                    if member.name.endswith(".data"):
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        for line in f:
                            try:
                                tweet = json.loads(line.decode("utf-8", errors="replace"))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue

                            if "in_reply_to" in tweet or "geo" not in tweet:
                                # Discard retweets and non geolocated
                                continue

                            lon = tweet["geo"].get("longitude")
                            lat = tweet["geo"].get("latitude")

                            geo_tweets.append({
                                    "tweetId": str(tweet.get("id")) if tweet.get("id") is not None else None,
                                    "body": tweet.get("tweet"),
                                    "postedTime": tweet.get("date"),
                                    "userId": str(tweet.get("user",dict()).get("id")) if tweet.get("user",dict()).get("id") is not None else None,
                                    "lon": lon,
                                    "lat": lat
                                })                       

    # Convert to Polars DataFrame
    df_geo = pl.DataFrame(geo_tweets, schema=schema)
    print(f"Collected {df_geo.height} geolocated tweets")
    return df_geo

def get_france_mainland_with_islands():
    """Returns multipolygon of France mainland and closeby islands (lon-lat crs)."""

    # Obtain France geometry (EPSG:4326)
    france = ox.geocode_to_gdf("France")
    france_geom = france.geometry.iloc[0]

    # Extract mainland (largest polygon)
    mainland_geom = max(france_geom.geoms, key=lambda g: g.area)
    france_mainland = gpd.GeoDataFrame(
        geometry=[mainland_geom],
        crs=france.crs
    )

    # Buffer mainland (e.g. 100 km) to retain nearby islands
    france_mainland_3035 = france_mainland.to_crs(EUROPEAN_PROJ)
    buffered_mainland_3035 = france_mainland_3035.buffer(100_000)
    buffered_mainland = buffered_mainland_3035.to_crs(france.crs).geometry.iloc[0]
    mainland_with_islands = france_geom.intersection(buffered_mainland)

    print("Num geometries france mainland with islands:", len(mainland_with_islands.geoms))
    return mainland_with_islands

def detect_home_location(gdf_geo):
    """Returns dataframe with home location by user."""

    # Project
    gdf_geo = gdf_geo.to_crs(EUROPEAN_PROJ)

    # Extract x, y
    gdf_geo["x"] = gdf_geo["geometry"].apply(lambda point: point.x)
    gdf_geo["y"] = gdf_geo["geometry"].apply(lambda point: point.y)

    # Snap to lower-left corner of 100m grid cell
    gdf_geo["x_100m"] = np.floor(gdf_geo["x"] / 100) * 100
    gdf_geo["y_100m"] = np.floor(gdf_geo["y"] / 100) * 100

    # For each user, find most common location
    location_counts = (
        gdf_geo
        .groupby(['userId', 'x_100m', 'y_100m'])
        .size()
        .reset_index(name='count')
    )
    idx = location_counts.groupby('userId')['count'].idxmax()
    user_home_locations = location_counts.loc[idx, ['userId', 'x_100m', 'y_100m', 'count']]
    user_home_locations = user_home_locations.rename(columns={
        'x_100m': 'x_home', 
        'y_100m': 'y_home',
        'count': 'home_tweet_count'
    })

    return user_home_locations

def main():

    collect = False
    if collect:

        # Collect geo tweets
        df_geo = collect_geo_located_tweets(DATA_PATH)

        # Save
        df_geo.write_parquet(
            f"{SAVE_DATA_PATH}snapshot_geo_tweets.parquet",
            compression="zstd"
        )

    # Read
    df_geo = pl.read_parquet(
        f"{SAVE_DATA_PATH}snapshot_geo_tweets.parquet")
    print("Num geo located tweets:", df_geo.height)
    print("Num geo located users:", df_geo["userId"].unique().len())

    # Filter out locations appearing more than 500 times
    df_valid_lonlat = (
        df_geo
        .drop_nulls(subset=["lon","lat"])
        .group_by(["lon","lat"]).agg(pl.len().alias("count"))
        .filter(pl.col("count")<=500)
    )
    df_geo = df_geo.join(
        df_valid_lonlat, on=["lon","lat"], how="right"
    ).select(df_geo.columns)
    print("Num geo located tweets <=500:", df_geo.height)
    print("Num geo located users <=500:", df_geo["userId"].unique().len())

    # Convert to gdf
    df_temp = df_geo.to_pandas()
    gdf_geo = gpd.GeoDataFrame(
        df_temp,
        geometry=gpd.points_from_xy(
            df_temp["lon"], df_temp["lat"], crs=LON_LAT_PROJ
            ),
        crs=LON_LAT_PROJ
    )
    
    # Get France mainland with closeby islands
    mainland_with_islands = get_france_mainland_with_islands()

    # Filter tweets within France
    gdf_geo = gdf_geo.sjoin(
        gpd.GeoDataFrame(
            [mainland_with_islands],
            columns=["geometry"], geometry="geometry", 
            crs=LON_LAT_PROJ
            ),
        predicate="within",
        how="inner"
        ).drop(columns="index_right")
    print("Num tweets in France:", gdf_geo.shape[0])
    print("Num users in France:", len(gdf_geo["userId"].unique()))

    # Detect home location
    user_home_locations = detect_home_location(gdf_geo)
    print("Num home inferred users:", user_home_locations.shape[0])

    # Save
    user_home_locations.to_parquet(
        f"{SAVE_DATA_PATH}user_home_locations.parquet",
        engine="pyarrow", 
        index=False
    )

    # Project to lon lat and save
    user_home_locations_lonlat = user_home_locations.copy()
    user_home_locations_lonlat["geometry"] = gpd.points_from_xy(
        user_home_locations_lonlat["x_home"], user_home_locations_lonlat["y_home"],
        crs=EUROPEAN_PROJ
        ).to_crs(LON_LAT_PROJ)
    user_home_locations_lonlat["lon"] = user_home_locations_lonlat["geometry"].x
    user_home_locations_lonlat["lat"] = user_home_locations_lonlat["geometry"].y
    user_home_locations_lonlat = user_home_locations_lonlat.drop(columns=["x_home","y_home","geometry"])
    user_home_locations_lonlat.to_parquet(
        f"{SAVE_DATA_PATH}user_home_locations_lonlat.parquet",
        engine="pyarrow", 
        index=False
    )

if __name__ == "__main__":
    main()