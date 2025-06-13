import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------------------

#Preprocessing the dataset

def load_and_preprocess_data():
    df = pd.read_csv('dataset/household_power_consumption.txt', 
                 sep=';',
                 parse_dates={'datetime': ['Date', 'Time']}, 
                 na_values='?',
                 infer_datetime_format=True, 
                 low_memory=False)
    df= df.dropna()
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df = df.set_index('datetime')
    return df

#---------------------------------------------------------------------------------------------------------------------
#What was the average active power consumption in March 2007?

def march_2007_data(df):
    march_2007 = df.loc['2007-03']
    print(f"Data for March 2007:\n{march_2007}")
    avg_power_march = march_2007['Global_active_power'].mean()
    print("Average active power consumption in March 2007:",avg_power_march," kW")

#---------------------------------------------------------------------------------------------------------------------
#What hour of the day had the highest power usage on Christmas 2006?

def christmas_2006_data(df):
    christmas_2006 = df.loc['2006-12-25']
    print(f"Data for Christmas 2006:\n{christmas_2006}")
    hourly_usage = christmas_2006.groupby(christmas_2006.index.hour)['Global_active_power'].mean()
    max_hour = hourly_usage.idxmax()
    print(f"The hour of the day with the highest power usage on Christmas 2006: {max_hour}:00")

#---------------------------------------------------------------------------------------------------------------------
#Compare energy usage (Global_active_power) on weekdays vs weekends.

def weekdays_weekends_data(df):
    # Create a column to identify weekdays (0-4) vs weekends (5-6)
    df['isWeekend'] = df.index.dayofweek>=5
    # Compare average power
    weekday_avg = df[~df['isWeekend']]['Global_active_power'].mean()
    weekend_avg = df[df['isWeekend']]['Global_active_power'].mean()

    print(f"Weekday average power: {weekday_avg:.2f} kW")
    print(f"Weekend average power: {weekend_avg:.2f} kW")

#---------------------------------------------------------------------------------------------------------------------
#Find days where energy consumption exceeded 5 kWh.

def energy_exceeding_5kWh(df):
    # Calculate daily average power
    exceeded_days = df.resample('D')['Global_active_power'].sum()[df.resample('D')['Global_active_power'].sum() > 5]
    print("Days where energy consumption exceeded 5 kWh:")
    print(exceeded_days)

#---------------------------------------------------------------------------------------------------------------------
#Plot the energy usage trend for the first week of January 2007.

def plot_energy_usage_jan_2007(df):
    # Filter data for the first week of January 2007
    first_week_jan_2007 = df[(df.index.year == 2007) & (df.index.month == 1) & (df.index.day <= 7)]

    # Plot the energy usage trend
    plt.figure(figsize=(10,6))
    plt.plot(first_week_jan_2007.index, first_week_jan_2007['Global_active_power'])
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.title('Energy Usage Trend for the First Week of January 2007')
    plt.show()

#---------------------------------------------------------------------------------------------------------------------
#Find the average voltage for each day of the first week of February 2007.

def avg_voltage_firstweek_feb_2007(df):
    first_week_feb= df['2007-02-01':'2007-02-07']
    daily_voltage = first_week_feb['Voltage'].resample('D').mean()
    print("Average voltage for each day of the first week of February 2007:")
    print(daily_voltage)

#---------------------------------------------------------------------------------------------------------------------
#What is the correlation between global active power and sub-metering values?

def correlation_global_active_power_sub_metering(df):
    sub_meters = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    correlation = df[['Global_active_power'] + sub_meters].corr()
    print("Correlation between global active power and sub-metering values:")
    print(correlation)
    print("\n\n",correlation['Global_active_power'][sub_meters])
#---------------------------------------------------------------------------------------------------------------------
# Main function
if __name__ == "__main__":
    df = load_and_preprocess_data()

    march_2007_data(df)
    #christmas_2006_data(df)
    #weekdays_weekends_data(df)
    #energy_exceeding_5kWh(df)
    #plot_energy_usage_jan_2007(df)
    #avg_voltage_firstweek_feb_2007(df)
    #correlation_global_active_power_sub_metering(df)