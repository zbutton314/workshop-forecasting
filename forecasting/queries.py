import sqlite3
import pandas as pd

from forecasting.utils import *


c = Clock()


def get_ride_austin_data(filepath):
    c.start()
    query = '''
        WITH rides AS (
            SELECT
                 a.RIDE_ID                                      AS ride_id
                ,DATE(created_date)                             AS date
                ,SUBSTR(created_date, 11, 2)                    AS hour
                ,distance_travelled                             AS distance
                ,start_location_lat                             AS start_lat
                ,start_location_long                            AS start_long
                ,end_location_lat                               AS end_lat
                ,end_location_long                              AS end_long
                ,round_up_amount                                AS charity_amount
                ,(CASE
                    WHEN requested_car_category = "b'LUXURY'"
                        THEN "b'PREMIUM'"
                    ELSE requested_car_category
                  END)                                          AS car_category
                ,make                                           AS car_make
                ,model                                          AS car_model
                ,HOURLYVISIBILITY                               AS hourly_visibility
                ,HOURLYDRYBULBTEMPC                             AS hourly_temp
                ,HOURLYRelativeHumidity                         AS hourly_humidity
                ,HOURLYWindSpeed                                AS hourly_wind_speed
                ,HOURLYWindDirection                            AS hourly_wind_direction
                ,HOURLYPrecip                                   AS hourly_precip
            FROM rides_a a
                INNER JOIN rides_b b
                    ON a.RIDE_ID = b.RIDE_ID
                INNER JOIN weather w
                    ON a.RIDE_ID = w.RIDE_ID
            WHERE requested_car_category <> "b'HONDA'"
                AND status = "b'DISPATCHED'"
        ),
        -- NOTE:
        -- This will only contain weather for hours in which a ride occurred.
        hourly_weather AS (
            SELECT
                 date
                ,hour
                ,MAX(hourly_visibility) AS visibility
                ,MAX(hourly_temp) AS temp
                ,MAX(hourly_humidity) AS humidity
                ,MAX(hourly_wind_speed) AS wind_speed
                ,MAX(hourly_wind_direction) AS wind_direction
                ,MAX(hourly_precip) AS precip
            FROM rides
            GROUP BY date, hour
        ),
        daily_weather AS (
            SELECT
                 date
                ,AVG(visibility) AS visibility
                ,MAX(temp)       AS temp_max
                ,MIN(temp)       AS temp_min
                ,AVG(humidity)   AS humidity
                ,AVG(wind_speed) AS wind_speed
                ,SUM(precip)     AS precip
            FROM hourly_weather
            GROUP BY date
        )
        SELECT
             r.ride_id
            ,r.date
            ,r.car_category
            ,r.distance
            ,r.charity_amount
            ,dw.visibility    AS daily_visibility
            ,dw.temp_max      AS daily_temp_max
            ,dw.temp_min      AS daily_temp_min
            ,dw.humidity      AS daily_humidity
            ,dw.wind_speed    AS daily_wind_speed
            ,dw.precip        AS daily_precip
        FROM rides r
            INNER JOIN daily_weather dw
                ON r.date = dw.date
    '''
    conn = sqlite3.connect(filepath)
    df = pd.read_sql_query(query, conn)
    conn.close()
    et = c.stop()
    print(f'Imported Rides Data: {df.shape[0]} rows, {et} seconds')

    return df
