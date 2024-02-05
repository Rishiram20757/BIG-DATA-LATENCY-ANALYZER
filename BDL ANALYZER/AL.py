import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import haversine_distances

df1 = pd.read_csv('100_vehicles_raw_data.csv')
df2 = pd.read_csv('TEST.csv')

df = pd.concat([df1, df2], axis=0, ignore_index=True)
df.head(10)

df = df[df['IGNITION_STATUS'] != 0]
df = df[df['LATITUDE'] != 0]
df = df[df['LONGITUDE'] != 0]
df['EVENT_LOCAL'] = pd.to_datetime(df['EVENT_LOCAL'], dayfirst=True)
df['PUBLISH_UTC'] = pd.to_datetime(df['PUBLISH_UTC'], dayfirst=True)
df['RECEIVED_TIMESTAMP'] = pd.to_datetime(df['RECEIVED_TIMESTAMP'], dayfirst=True)
df['PUBLISED LOCAL'] = pd.to_datetime(df['PUBLISED LOCAL'], dayfirst=True)
df['RECEIVED LOCAL'] = pd.to_datetime(df['RECEIVED LOCAL'], dayfirst=True)
# --------------------------------------------------------------------------------------------------#
#                                           CORRELATION                                            #
# --------------------------------------------------------------------------------------------------#
datetime_columns = df.select_dtypes(include=['datetime64']).columns
if len(datetime_columns) >= 2:

    correlations_all_datetime = df[datetime_columns].corr()
    print("Correlation among all datetime columns:\n")
    print(correlations_all_datetime)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations_all_datetime, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Basic Correlation Heatmap of Datetime Columns')
    plt.show()
    selected_column = 'RECEIVED LOCAL'

    # Select other datetime columns for correlation
    datetime_columns_selected = df[datetime_columns].columns.difference([selected_column])

    if len(datetime_columns_selected) >= 1:
        # Calculate correlations between the selected datetime column and others
        correlations_selected_column = df[[selected_column] + list(datetime_columns_selected)].corr()[selected_column]

    # Display the correlation table
    print(f"Correlation of '{selected_column}' with other datetime columns:\n")
    print(correlations_selected_column)

    # Plot bar chart for visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations_selected_column.index, y=correlations_selected_column.values)
    plt.title(f'Correlation of {selected_column} with Other Datetime Columns')
    plt.xlabel('Datetime Columns')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.show()
else:
    selected_column = 'RECEIVED LOCAL'
    print(f"There are not enough other datetime columns for correlation with '{selected_column}'.")

# --------------------------------------------------------------------------------------------------#
#                                             LATENCY                                              #
# --------------------------------------------------------------------------------------------------#
df['Recieving Latency'] = pd.to_datetime(df['Recieving Latency'], dayfirst=True)
df['Publishing Latency'] = pd.to_datetime(df['Publishing Latency'], dayfirst=True)
df['Recieving Latency'] = df['RECEIVED LOCAL'] - df['EVENT_LOCAL']
df['Publishing Latency'] = df['PUBLISED LOCAL'] - df['EVENT_LOCAL']

print(df.info())
# --------------------------------------------------------------------------------------------------#
#                               LINEAR REGRESSION GRAPHS                                           #
# --------------------------------------------------------------------------------------------------#
datetime_columns = df.select_dtypes(include=['datetime64']).columns
if len(datetime_columns) >= 1:
    for column in datetime_columns:
        if column != selected_column:
            plt.figure(figsize=(8, 6))
            x_values = pd.to_numeric(df[column])
            y_values = pd.to_numeric(df[selected_column])
            sns.regplot(x=x_values, y=y_values, scatter_kws={'s': 50}, ci=None)
            plt.title(f'Linear Regression: {column} vs {selected_column}')
            plt.xlabel(column)
            plt.ylabel(selected_column)
            plt.show()
else:
    print("There are not enough datetime columns for linear regression.")
# --------------------------------------------------------------------------------------------------#
#                                          BINNING                                                 #
# --------------------------------------------------------------------------------------------------#
column1 = 'Recieving Latency'
column2 = 'RECEIVED LOCAL'
custom_bins = [pd.to_timedelta('00:00:00'),
               pd.to_timedelta('00:00:01'), pd.to_timedelta('00:01:00'), pd.to_timedelta('00:02:00'),
               pd.to_timedelta('00:03:00'), pd.to_timedelta('00:04:00'), pd.to_timedelta('00:05:00'),
               pd.to_timedelta('00:06:00'), pd.to_timedelta('00:10:00'), pd.to_timedelta('00:30:00'),
               pd.to_timedelta('01:00:00'), pd.to_timedelta('01:30:00'), pd.to_timedelta('02:00:00'),
               pd.to_timedelta('02:30:00'), pd.to_timedelta('03:00:00'), pd.to_timedelta('03:30:00'),
               pd.to_timedelta('04:00:00'), pd.to_timedelta('05:00:00'), pd.to_timedelta('06:00:00'),
               pd.to_timedelta('07:00:00'), pd.to_timedelta('08:00:00'), pd.to_timedelta('09:00:00'),
               pd.to_timedelta('10:00:00'), pd.to_timedelta('11:00:00'), pd.to_timedelta('12:00:00'),
               pd.to_timedelta('13:00:00'), pd.to_timedelta('14:00:00'), pd.to_timedelta('15:00:00'),
               pd.to_timedelta('16:00:00'), pd.to_timedelta('17:00:00'), pd.to_timedelta('18:00:00'),
               pd.to_timedelta('19:00:00'), pd.to_timedelta('20:00:00'), pd.to_timedelta('21:00:00'),
               pd.to_timedelta('22:00:00'), pd.to_timedelta('23:00:00'), pd.to_timedelta('24:00:00'),
               pd.to_timedelta('25:00:00'), pd.to_timedelta('26:00:00'), pd.to_timedelta('27:00:00'),
               pd.to_timedelta('28:00:00'), pd.to_timedelta('29:00:00'), pd.to_timedelta('30:00:00'),
               pd.to_timedelta('32:00:00'), pd.to_timedelta('34:00:00'), pd.to_timedelta('36:00:00'),
               pd.to_timedelta('38:00:00'), pd.to_timedelta('40:00:00'), pd.to_timedelta('48:00:00'),
               pd.to_timedelta('999:00:00')]
bins = pd.cut(df[column1].astype(str).apply(lambda x: pd.to_timedelta(x)), bins=custom_bins,
              right=False)
bin_percentage = df.groupby(bins, observed=False)[column2].count() / len(df) * 100
data_table = pd.DataFrame({
    'Bin': bin_percentage.index,
    'Percentage': bin_percentage.values
}).dropna()
plt.figure(figsize=(10, 6))
sns.barplot(x='Bin', y='Percentage', data=data_table)
plt.title(f'Percentage Distribution in Custom Bins: {column1} vs {column2}')
plt.xlabel(column1)
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()
print(data_table)

#--------------------Lines - Publishing vs recieving latencies
sns.set(style="whitegrid", palette="light:#85C1E9")
publish_color = 'green'
event_color = 'purple'
plt.figure(figsize=(10, 6))
sns.lineplot(x='PUBLISED LOCAL', y='RECEIVED LOCAL', data=df, marker='o', markersize=10, color=publish_color, label='RECEIVED LOCAL')
sns.lineplot(x='PUBLISED LOCAL', y='PUBLISED LOCAL', data=df, marker='o', markersize=10, color=event_color, label='PUBLISED LOCAL')
plt.xlabel('Event Local Time')
plt.ylabel('Time')
plt.title('Comparison of Publish Local and Received Local Times')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#public and event local diff vs event local

sns.set(style="whitegrid", palette="light:#85C1E9")
publish_color = 'green'
event_color = 'purple'

plt.figure(figsize=(10, 6))
sns.lineplot(x='EVENT_LOCAL', y='PUBLISED LOCAL', data=df, marker='o', markersize=10, color=publish_color, label='PUBLISED LOCAL')
sns.lineplot(x='EVENT_LOCAL', y='EVENT_LOCAL', data=df, marker='o', markersize=10, color=event_color, label='EVENT_LOCAL')
plt.xlabel('Event Local Time')
plt.ylabel('Time')
plt.title('Comparison of Event Local and Publish Local Times')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#longlat vs recieved
fig = px.scatter_mapbox(df,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='RECEIVED LOCAL',
                        size_max=20,
                        zoom=4,
                        title='Scatter Plot of Latitude, Longitude, and Received Local')
fig.update_layout(mapbox_style="open-street-map")
fig.show()

#-----------------------------CLUSTERED NODES---------------------------------#
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
import plotly.express as px

df_cluster = df[['LATITUDE', 'LONGITUDE', 'RECEIVED LOCAL', 'VIN']]
df_cluster = df_cluster.dropna()
le = LabelEncoder()
df_cluster['Vehicle Label'] = le.fit_transform(df_cluster['VIN'])
X = df_cluster[['LATITUDE', 'LONGITUDE', 'Vehicle Label']]
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_cluster['Cluster Label'] = kmeans.fit_predict(X)
custom_colors = ['white','cyan', 'blue', 'yellow', 'orange', 'red']
fig = px.scatter_mapbox(df_cluster,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='Cluster Label',
                        size_max=20,
                        color_continuous_scale=custom_colors,  # Reusing custom color sequence
                        zoom=4,
                        title='Clustered NODES',
                        labels={'Cluster Label': 'Cluster'})
fig.update_layout(mapbox_style="open-street-map")
fig.show()

#---------------------------------------ROUTE MAP-------------------------------------------#
df_cluster = df[['LATITUDE', 'LONGITUDE', 'RECEIVED LOCAL', 'VIN',]]
df_cluster = df_cluster.dropna()
le = LabelEncoder()
df_cluster['Vehicle Label'] = le.fit_transform(df_cluster['VIN'])
X = df_cluster[['LATITUDE', 'LONGITUDE', 'Vehicle Label']]
n_clusters = 10
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
df_cluster['Cluster Label'] = kmeans.fit_predict(X)
fig = px.scatter_mapbox(df_cluster,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='VIN',  # Use 'VIN' as the color parameter
                        size_max=20,
                        zoom=4,
                        title='VEHICLES ROUTE',
                        labels={'Cluster Label': 'Cluster'})

fig.update_layout(mapbox_style="open-street-map")
fig.show()

#------------------------------------LATENCY IN MAP--------------------#
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
df_cluster = df[['LATITUDE', 'LONGITUDE', 'RECEIVED LOCAL', 'VIN', 'Recieving Latency']]
df_cluster['Receiving Latency Seconds'] = df_cluster['Recieving Latency'].dt.total_seconds()
scaler = MinMaxScaler(feature_range=(5, 20))
df_cluster['Normalized Size'] = scaler.fit_transform(df_cluster[['Receiving Latency Seconds']])
scattermapbox_trace = go.Scattermapbox(
    lat=df_cluster['LATITUDE'],
    lon=df_cluster['LONGITUDE'],
    mode='markers',
    marker=dict(
        size=df_cluster['Normalized Size'],
        color=df_cluster['Receiving Latency Seconds'],
        colorscale=['white','cyan', 'blue', 'yellow', 'orange', 'pink', 'red'],
        cmin=df_cluster['Receiving Latency Seconds'].min(),
        cmax=df_cluster['Receiving Latency Seconds'].max(),
        colorbar=dict(title='Receiving Latency Seconds'),
    ),
    text=df_cluster['Receiving Latency Seconds'].astype(str) + ' seconds',
)
layout = go.Layout(
    title='Location vs Receiving Latency',
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=df_cluster['LATITUDE'].mean(), lon=df_cluster['LONGITUDE'].mean()),
        zoom=4,
    ),
)
fig = go.Figure(data=[scattermapbox_trace], layout=layout)
fig.show()

#--------------------------------------DBSCAN CLUSTERING------------------------#
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['LATITUDE', 'LONGITUDE']])
radius_km = 1
epsilon = radius_km / 111.32
dbscan = DBSCAN(eps=epsilon, min_samples=3)
df_cluster['DBSCAN Cluster Label'] = dbscan.fit_predict(X_scaled)
custom_colors = ['white','cyan', 'blue', 'yellow', 'orange', 'red']
fig = px.scatter_mapbox(df_cluster,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='DBSCAN Cluster Label',
                        size_max=20,
                        color_continuous_scale=custom_colors,  # Custom color sequence
                        zoom=4,
                        title='DBSCAN Clustered NODES',
                        labels={'DBSCAN Cluster Label': 'MIN_pts'})

fig.update_layout(mapbox_style="open-street-map")
fig.show()

import plotly.express as px
df_cluster['Receiving Latency Seconds'] = df_cluster['Recieving Latency'].dt.total_seconds()
df_cluster_filtered = df_cluster[df_cluster['DBSCAN Cluster Label'] != -1]
custom_colors = ['white','cyan', 'blue', 'yellow', 'orange', 'red']
fig = px.scatter_mapbox(df_cluster_filtered,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='Receiving Latency Seconds',
                        size_max=20,
                        color_continuous_scale=custom_colors,  # Custom color sequence
                        zoom=4,
                        title='Nodes Colored by Receiving Latency [DBSCAN]',
                        labels={'Receiving Latency Seconds': 'Latency (seconds)'})

fig.update_layout(mapbox_style="open-street-map")
fig.show()

#----------------------------------------LATENCY MORE THAN 20K---------------------------------#
df_cluster['Receiving Latency Seconds'] = df_cluster['Recieving Latency'].dt.total_seconds()
df_filtered = df_cluster[(df_cluster['DBSCAN Cluster Label'] != -1) & (df_cluster['Receiving Latency Seconds'] > 3600)]
custom_colors = ['cyan', 'blue', 'yellow', 'orange', 'red']
fig = px.scatter_mapbox(df_filtered,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='Receiving Latency Seconds',  # Color based on latency
                        size_max=20,
                        color_continuous_scale=custom_colors,  # Custom color sequence
                        zoom=4,
                        title='Latency More than 1 hour (3600 seconds)',
                        labels={'Receiving Latency Seconds': 'Latency (seconds)'},
                        hover_data={'VIN': True })

fig.update_layout(mapbox_style="open-street-map")
fig.show()
import plotly.express as px
import plotly.graph_objects as go
df['Receiving Latency Seconds'] = df['Recieving Latency'].dt.total_seconds()
custom_colors = ['white', 'cyan', 'blue', 'yellow', 'orange', 'red']
color_map = {'active': 'green', 'Permanent Disconnection': 'red', 'other_status': 'gray'}
df['Primary TSP Color'] = df['Primary Status'].map(color_map)
df['Fallback TSP Color'] = df['Fallback Status'].map(color_map)
df_filtered = df[df['Receiving Latency Seconds'] >= 20000]
table_data = df_filtered[['Receiving Latency Seconds', 'Primary TSP', 'Fallback TSP']]
table_data['SP Name'] = table_data['Primary TSP'].where(df_filtered['Primary Status'] == 'active', df_filtered['Fallback TSP'])
table_data = table_data[['Receiving Latency Seconds', 'SP Name']].sort_values(by='Receiving Latency Seconds', ascending=False)
print('\nLatency and SP Name Table:')
print(table_data)
fig = px.scatter_mapbox(df,
                        lat='LATITUDE',
                        lon='LONGITUDE',
                        color='Receiving Latency Seconds',  # Color based on latency
                        size_max=20,
                        hover_name='VIN',
                        hover_data=['Primary TSP', 'Fallback TSP', 'Primary Status', 'Fallback Status'],
                        color_continuous_scale=custom_colors,
                        title='Latency Analysis with Service Providers',
                        labels={'Receiving Latency Seconds': 'Latency (seconds)'})

fig.update_layout(mapbox_style="open-street-map")
fig.show()
##########################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

column1 = 'Recieving Latency'
column2 = 'RECEIVED LOCAL'
column3 = 'Merged TSP'

custom_bins = [pd.to_timedelta('00:00:00'), pd.to_timedelta('00:00:01'), pd.to_timedelta('00:01:00'),
               pd.to_timedelta('00:02:00'), pd.to_timedelta('00:03:00'), pd.to_timedelta('00:04:00'),
               pd.to_timedelta('00:05:00'), pd.to_timedelta('00:06:00'), pd.to_timedelta('00:10:00'),
               pd.to_timedelta('00:30:00'), pd.to_timedelta('01:00:00'), pd.to_timedelta('01:30:00'),
               pd.to_timedelta('02:00:00'), pd.to_timedelta('02:30:00'), pd.to_timedelta('03:00:00'),
               pd.to_timedelta('03:30:00'), pd.to_timedelta('04:00:00'), pd.to_timedelta('05:00:00'),
               pd.to_timedelta('06:00:00'), pd.to_timedelta('07:00:00'), pd.to_timedelta('08:00:00'),
               pd.to_timedelta('09:00:00'), pd.to_timedelta('10:00:00'), pd.to_timedelta('11:00:00'),
               pd.to_timedelta('12:00:00'), pd.to_timedelta('13:00:00'), pd.to_timedelta('14:00:00'),
               pd.to_timedelta('15:00:00'), pd.to_timedelta('16:00:00'), pd.to_timedelta('17:00:00'),
               pd.to_timedelta('18:00:00'), pd.to_timedelta('19:00:00'), pd.to_timedelta('20:00:00'),
               pd.to_timedelta('21:00:00'), pd.to_timedelta('22:00:00'), pd.to_timedelta('23:00:00'),
               pd.to_timedelta('24:00:00'), pd.to_timedelta('25:00:00'), pd.to_timedelta('26:00:00'),
               pd.to_timedelta('27:00:00'), pd.to_timedelta('28:00:00'), pd.to_timedelta('29:00:00'),
               pd.to_timedelta('30:00:00'), pd.to_timedelta('32:00:00'), pd.to_timedelta('34:00:00'),
               pd.to_timedelta('36:00:00'), pd.to_timedelta('38:00:00'), pd.to_timedelta('40:00:00'),
               pd.to_timedelta('48:00:00'), pd.to_timedelta('999:00:00')]
bins = pd.cut(df[column1].astype(str).apply(lambda x: pd.to_timedelta(x)), bins=custom_bins, right=False)
bin_percentage = df.groupby([bins, column3], observed=False)[column2].count() / len(df) * 100
data_table = pd.DataFrame({
    'Bin': bin_percentage.index.get_level_values(0),
    column3: bin_percentage.index.get_level_values(1),
    column2: bin_percentage.values
}).dropna()
plt.figure(figsize=(12, 6))
bar2 = sns.barplot(x='Bin', y=column2, hue=column3, data=data_table, dodge=True)
legend_labels = {'AIRTEL', 'RECEIVED LOCAL', 'BOTH', 'BSNL'}
handles, _ = bar2.get_legend_handles_labels()
plt.title(f'Double Bar Graph: {column1} vs {column3}')
plt.xlabel(column1)
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()

#########################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

column1 = 'Recieving Latency'
column2 = 'RECEIVED LOCAL'
column3 = 'Merged TSP'
custom_bins = [pd.to_timedelta('00:00:00'), pd.to_timedelta('00:00:01'), pd.to_timedelta('00:01:00'),
               pd.to_timedelta('00:02:00'), pd.to_timedelta('00:03:00'), pd.to_timedelta('00:04:00'),
               pd.to_timedelta('00:05:00'), pd.to_timedelta('00:06:00'), pd.to_timedelta('00:10:00'),
               pd.to_timedelta('00:30:00'), pd.to_timedelta('01:00:00'), pd.to_timedelta('01:30:00'),
               pd.to_timedelta('02:00:00'), pd.to_timedelta('02:30:00'), pd.to_timedelta('03:00:00'),
               pd.to_timedelta('03:30:00'), pd.to_timedelta('04:00:00'), pd.to_timedelta('05:00:00'),
               pd.to_timedelta('06:00:00'), pd.to_timedelta('07:00:00'), pd.to_timedelta('08:00:00'),
               pd.to_timedelta('09:00:00'), pd.to_timedelta('10:00:00'), pd.to_timedelta('11:00:00'),
               pd.to_timedelta('12:00:00'), pd.to_timedelta('13:00:00'), pd.to_timedelta('14:00:00'),
               pd.to_timedelta('15:00:00'), pd.to_timedelta('16:00:00'), pd.to_timedelta('17:00:00'),
               pd.to_timedelta('18:00:00'), pd.to_timedelta('19:00:00'), pd.to_timedelta('20:00:00'),
               pd.to_timedelta('21:00:00'), pd.to_timedelta('22:00:00'), pd.to_timedelta('23:00:00'),
               pd.to_timedelta('24:00:00'), pd.to_timedelta('25:00:00'), pd.to_timedelta('26:00:00'),
               pd.to_timedelta('27:00:00'), pd.to_timedelta('28:00:00'), pd.to_timedelta('29:00:00'),
               pd.to_timedelta('30:00:00'), pd.to_timedelta('32:00:00'), pd.to_timedelta('34:00:00'),
               pd.to_timedelta('36:00:00'), pd.to_timedelta('38:00:00'), pd.to_timedelta('40:00:00'),
               pd.to_timedelta('48:00:00'), pd.to_timedelta('999:00:00')]
bins = pd.cut(df[column1].astype(str).apply(lambda x: pd.to_timedelta(x)), bins=custom_bins, right=False)
bin_count = df.groupby([bins, column3])[column1].count().reset_index(name='Count')
data_table = pd.DataFrame({
    'Bin': bin_count.iloc[:, 0],
    column3: bin_count.iloc[:, 1],
    'Count': bin_count['Count']
}).dropna()

############################
plt.figure(figsize=(12, 6))
bar2 = sns.barplot(x='Bin', y='Count', hue=column3, data=data_table, dodge=True)
plt.title(f'Double Bar Graph: {column1} vs {column3}')
plt.xlabel(column1)
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title=column3)
plt.show()
airtel_table = data_table[data_table[column3] == 'AIRTEL'].reset_index(drop=True)
both_table = data_table[data_table[column3] == 'BOTH'].reset_index(drop=True)
both_table['Bin'] = both_table['Bin'].astype(str)
airtel_table['Bin'] = airtel_table['Bin'].astype(str)
fig_both = px.bar(both_table, x='Bin', y='Count', color='Bin', title=f'Bar Graph: {column1} vs BOTH')
fig_both.update_xaxes(title_text=column1)
fig_both.update_yaxes(title_text='Count')
fig_both.update_layout(xaxis=dict(tickangle=45))
fig_airtel = px.bar(airtel_table, x='Bin', y='Count', color='Bin', title=f'Bar Graph: {column1} vs AIRTEL')
fig_airtel.update_xaxes(title_text=column1)
fig_airtel.update_yaxes(title_text='Count')
fig_airtel.update_layout(xaxis=dict(tickangle=45))
fig_both.update_traces(xaxis='x', hoverinfo='x+y')
fig_airtel.update_traces(xaxis='x', hoverinfo='x+y')
fig_both.show()
fig_airtel.show()
print(airtel_table)
print(both_table)
###############################################
"Red zone creation"
from math import radians, sin, cos, sqrt, atan2
red_zone_location = (23.495544, 78.962144)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
df['RedZone'] = df.apply(lambda row: haversine(row['LATITUDE'], row['LONGITUDE'], red_zone_location[0], red_zone_location[1]) <= 200, axis=1)
print(df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
column1 = 'Recieving Latency'
column2 = 'RECEIVED LOCAL'
column3 = 'Merged TSP'
df_filtered = df[df['RedZone'] != True]
custom_bins = [pd.to_timedelta('00:00:00'), pd.to_timedelta('00:00:01'), pd.to_timedelta('00:01:00'),
               pd.to_timedelta('00:02:00'), pd.to_timedelta('00:03:00'), pd.to_timedelta('00:04:00'),
               pd.to_timedelta('00:05:00'), pd.to_timedelta('00:06:00'), pd.to_timedelta('00:10:00'),
               pd.to_timedelta('00:30:00'), pd.to_timedelta('01:00:00'), pd.to_timedelta('01:30:00'),
               pd.to_timedelta('02:00:00'), pd.to_timedelta('02:30:00'), pd.to_timedelta('03:00:00'),
               pd.to_timedelta('03:30:00'), pd.to_timedelta('04:00:00'), pd.to_timedelta('05:00:00'),
               pd.to_timedelta('06:00:00'), pd.to_timedelta('07:00:00'), pd.to_timedelta('08:00:00'),
               pd.to_timedelta('09:00:00'), pd.to_timedelta('10:00:00'), pd.to_timedelta('11:00:00'),
               pd.to_timedelta('12:00:00'), pd.to_timedelta('13:00:00'), pd.to_timedelta('14:00:00'),
               pd.to_timedelta('15:00:00'), pd.to_timedelta('16:00:00'), pd.to_timedelta('17:00:00'),
               pd.to_timedelta('18:00:00'), pd.to_timedelta('19:00:00'), pd.to_timedelta('20:00:00'),
               pd.to_timedelta('21:00:00'), pd.to_timedelta('22:00:00'), pd.to_timedelta('23:00:00'),
               pd.to_timedelta('24:00:00'), pd.to_timedelta('25:00:00'), pd.to_timedelta('26:00:00'),
               pd.to_timedelta('27:00:00'), pd.to_timedelta('28:00:00'), pd.to_timedelta('29:00:00'),
               pd.to_timedelta('30:00:00'), pd.to_timedelta('32:00:00'), pd.to_timedelta('34:00:00'),
               pd.to_timedelta('36:00:00'), pd.to_timedelta('38:00:00'), pd.to_timedelta('40:00:00'),
               pd.to_timedelta('48:00:00'), pd.to_timedelta('999:00:00')]

# Binning process
bins = pd.cut(df_filtered[column1].astype(str).apply(lambda x: pd.to_timedelta(x)), bins=custom_bins, right=False)
bin_count = df_filtered.groupby([bins, column3])[column1].count().reset_index(name='Count')
total_count = bin_count.groupby(column3)['Count'].transform('sum')
data_table = pd.DataFrame({
    'Bin': bin_count.iloc[:, 0],
    column3: bin_count.iloc[:, 1],
    'Count': bin_count['Count'],
    'Percentage': (bin_count['Count'] / total_count) * 100
}).dropna()
plt.figure(figsize=(12, 6))
bar2 = sns.barplot(x='Bin', y='Percentage', hue=column3, data=data_table, dodge=True)
plt.title(f'Double Bar Graph: {column1} vs {column3} (Excluding RedZone)')
plt.xlabel(column1)
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title=column3)
plt.show()
#-------------------------------------------REDZONE------------------------#
import pandas as pd
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
red_zone_location = (23.495544, 78.962144)
red_zone_radius_km = 150
def is_in_redzone(lat, lon):
    distance = haversine(lat, lon, *red_zone_location)
    return distance <= red_zone_radius_km
df_cluster['RedZone'] = df_cluster.apply(lambda row: is_in_redzone(row['LATITUDE'], row['LONGITUDE']), axis=1)
df_cluster['Receiving Latency Seconds'] = df_cluster['Recieving Latency'].dt.total_seconds()
scaler = MinMaxScaler(feature_range=(5, 20))
df_cluster['Normalized Size'] = scaler.fit_transform(df_cluster[['Receiving Latency Seconds']])
scattermapbox_trace = go.Scattermapbox(
    lat=df_cluster['LATITUDE'],
    lon=df_cluster['LONGITUDE'],
    mode='markers',
    marker=dict(
        size=df_cluster['Normalized Size'],
        color=df_cluster['Receiving Latency Seconds'],
        colorscale=['white', 'cyan', 'blue', 'yellow', 'orange', 'pink', 'red'],
        cmin=df_cluster['Receiving Latency Seconds'].min(),
        cmax=df_cluster['Receiving Latency Seconds'].max(),
        colorbar=dict(title='Receiving Latency Seconds'),
    ),
    text=df_cluster['Receiving Latency Seconds'].astype(str) + ' seconds',
    name='Nodes',  # Specify the name for the legend
)
redzone_trace = go.Scattermapbox(
    lat=df_cluster[df_cluster['RedZone']]['LATITUDE'],
    lon=df_cluster[df_cluster['RedZone']]['LONGITUDE'],
    mode='markers',
    marker=dict(
        size=10,
        color='red',
    ),
    text='RedZone',
    name='RedZone',
)
layout = go.Layout(
    title='Location vs Receiving Latency with RedZone',
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=df_cluster['LATITUDE'].mean(), lon=df_cluster['LONGITUDE'].mean()),
        zoom=4,
    ),
)
fig = go.Figure(data=[scattermapbox_trace, redzone_trace], layout=layout)
fig.show()
