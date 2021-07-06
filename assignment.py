"""
COMP1730/6730 S1 2020 - Project Assignment.

Author: <u6004244>
"""

import matplotlib.pyplot as plt
from assignment_helpers import plot_volumes
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import metrics


# Question_1
def read_dataset(filepath):
    """
    This function using Pandas read_csv to read the data
    and return the data with labels from the file
    """
    df = pd.read_csv(filepath)
    print(df.head())
    return df


print("\n[ Question_1 : Loaded Dataset ]\n")
df = read_dataset("assignment_lake_george_data.csv")


# Question_2_a
def largest_area(data):
    """
    This function using the numpy.max to return the largest area from the data
    """
    data[['area']].astype(float)
    area_data = data['area'].values
    max_area = np.max(area_data)
    print("Maximum Area of Lake : ", max_area, "Km^2")
    return max_area


print("\n[ Question_2_A ]\n")
largest_area(df)

def mean_area(data):
    data[['area']].astype(float)
    area_data = data['area'].values
    mean_area = np.mean(area_data)
    return mean_area


# Question_2_b
def average_volume(data):
    """
    This function using the numpy.mean return the average volume
    """
    volume_data = data['volume'].values
    average_volume = np.mean(volume_data)
    print("\n[ Question_2_B ] \n")
    print("Average Volume of the Lake : ", average_volume, 'Litres')
    return average_volume


average_volume(df)


# Question_2_c
def most_average_rainfall(data):
    """
    This function using the numpy.mean to give a average rainfall

    Parameters:
        average_value: the average rainfall value
        closest_rainfall_value: the value have the smallest difference with the average value
        df_target: the rows contains the closest_rainfall_value
        data_target: the date list of the target rows (rows contains the closest_rainfall_value)
    """
    rainfall_data = data['rainfall']
    average_value = np.mean(rainfall_data)
    # using lambda function to find the value with minimum differences compare with average_value
    closest_rainfall_value = min(rainfall_data, key=lambda x: abs(x - average_value))
    # print(closest_rainfall_value)
    # find the rows have closest_rainfall_value
    df_target = data.loc[data['rainfall'] == closest_rainfall_value]
    # find the date have rainfall closest to average
    data_target = list(df_target['date'].values)
    print("Year and Month has a rainfall closet to average :")
    for idx in range(len(data_target)):
        print(idx + 1, ", Year : ", str(data_target[idx])[:4], ", Month : ", str(data_target[idx])[-2:])


print("\n[ Question_2_C ]\n")
most_average_rainfall(df)


def hottest_month(data):
    """
    This function return the hottest month

    Parameters:
        target_data: data of max_temperature from the data file
        target_mean: the mean value of the maximum temperature in each month
        mean_temp_list: the list containing the mean maximum temperature from January to December (1-12)
        max_temp: the maximum temperature of each month's mean maximum temperature
    """
    month_data = []
    for idx in range(len(data)):
        month_data.append(str(data['date'][idx])[-2:])
    data['month'] = month_data
    month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    mean_temp_list = []
    # for each month, adding the mean maximum value into the list
    for str_month in month_list:
        df_target = data.loc[data['month'] == str_month]
        target_data = df_target['max_temperature'].values
        target_mean = np.mean(target_data)
        mean_temp_list.append(target_mean)
    # print(mean_temp_list)
    max_temp = np.max(mean_temp_list)
    # because the mean_temp_list contains the each month's mean max temperature,
    # after locate the maximum number, use the index to find the corresponding month from the month list
    for k in range(len(mean_temp_list)):
        if mean_temp_list[k] == max_temp:
            print(" Hottest month : ", month_list[k], ", Mean Temperature : ", max_temp)


print("\n[ Question_2_D ]\n")
hottest_month(df)


def get_year_month_column(data):
    """
    This function convert date column to Year-Month DateTime Format
    and return data
    """
    year_month_list = []
    for idx in range(len(data)):
        # using index to get the year and month
        temp_year = str(data['date'][idx])[:4]
        temp_month = str(data['date'][idx])[-2:]
        # print(temp_month)
        temp = temp_year + '-' + temp_month
        date_time_obj = datetime.datetime.strptime(temp, '%Y-%m')
        year_month_list.append(date_time_obj)
    data['year-month'] = year_month_list
    return data


def area_vs_volume_1(data):
    """
    This function plot the area against the volume
    """
    data = get_year_month_column(data)
    labels = ['areas(Km^2) * 1000 ', 'volumes(L)']
    #
    areas = data['area'] * 1000
    volumes = data['volume']
    colors = ['tomato', 'skyblue']
    x = data['year-month']
    y = np.vstack([areas, volumes])
    plt.title('[Question_3_1]  Area(Km^2) vs Volume(L)')
    plt.stackplot(x, y, labels=labels, colors=colors, edgecolor='black')
    plt.legend(loc=2)
    plt.show()


def area_vs_volume(data):
    """
    This function using add_axes() to plot the areas and volume as two sub-graphs
    """
    area_vs_volume_1(data)
    data = get_year_month_column(data)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1= fig.add_axes([0,0,1,1])
    # ax2 = fig.add_axes([0.25, 0.65, 0.5, 0.3])
    ax1.set_title('[Question_3_2]  Area(Km^2) vs Volume(L)')
    # ax2.set_title('Volume(L)')
    ax1.plot(data['year-month'], data['area'], color='green')
    ax2.plot(data['year-month'], data['volume'], color='blue')
    plt.show()


area_vs_volume(df)


def lake_george_simple_model(data, evaporation_rate):
    """
    This function using the equation of modelled_volume = modelled volume(n-1) +
    rainfall volume - evaporation volume return the modelled values
    Parameters:
        max_area: the largest area from the function largest_area(data)
        rainfall_volume: total rainfall volume of the largest area
        evapo_volume: total evaporation volume of the largest area
    """
    data = get_year_month_column(data)
    max_area = largest_area(data)
    # mean_a = mean_area(data)
    actual_volume = data['volume']
    rainfall_volume = data['rainfall'] * max_area
    evapo_volume = evaporation_rate * max_area
    # rainfall_volume = data['rainfall'] * mean_a
    # evapo_volume = evaporation_rate * mean_a

    modelled_volume = [0] * len(data)
    for idx in range(len(data)):
        if idx == 0:
            modelled_volume[0] = actual_volume[0]
        else:
            temp_volume = modelled_volume[idx - 1] + rainfall_volume[idx] - evapo_volume
            if temp_volume <= 0:
                pass
            else:
                modelled_volume[idx] = temp_volume
    return modelled_volume


plot_volumes(lake_george_simple_model(df, 55))


def lake_george_complex_model(data):
    """
    This function using the equation E = −3Tmin + 1.6Tmax − 2.5W + 4.5S − 0.4H to calculate the
    evaporation rate E, return the modelled volume
    Parameters:
        T: temperature (in Celsius)
        S: the solar exposure (in MJ/month/m2)
        W: the wind speed (in m/s)
        H: the humidity (as a percentage, i.e. as a number between 0 and 100)

    """
    data = get_year_month_column(data)
    actual_volume = data['volume']
    rainfall_volume = data['rainfall'] * data['area']
    T_min = data['min_temperature']
    T_max = data['max_temperature']
    W = data['wind_speed']
    S = data['solar_exposure']
    H = data['humidity']
    E = -3 * T_min + 1.6 * T_max - 2.5 * W + 4.5 * S - 0.4 * H
    evapo_volume = E * data['area']
    # modelled_volume = actual_volume + rainfall_volume - evapo_volume
    modelled_volume = [0] * len(data)
    for idx in range(len(data)):
        if idx == 0:
            modelled_volume[0] = actual_volume[0]
        else:
            temp_volume = modelled_volume[idx - 1] + rainfall_volume[idx] - evapo_volume[idx]
            if temp_volume <= 0:
                pass
            else:
                modelled_volume[idx] = temp_volume
    return modelled_volume


plot_volumes(lake_george_complex_model(df))


def evaluate_model(data, volumes):
    """
    This function measures the average magnitude of the errors
    Parameters:
        MAE: mean absolute error
        RMSE: root mean squared error
    """
    actual_volumes = data['volume']
    MAE = metrics.mean_absolute_error(actual_volumes, volumes)
    RMSE = np.sqrt(metrics.mean_squared_error(actual_volumes, volumes))
    print("\tMAE = ", MAE)
    print("\tRMSE = ", RMSE)


print("\n[Question_5 Evaluate Model :]\n")
simple_modelled_volume = lake_george_simple_model(df, 55)
complex_modelled_volume = lake_george_complex_model(df)

print("Evaluate Simple_Model : ")
evaluate_model(df, simple_modelled_volume)

print("Evaluate Complex_Model : ")
evaluate_model(df, complex_modelled_volume)
