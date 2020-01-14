#Evgeniya Zvigunova

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
import folium

# NEED TO COMMENT AND UNCOMMENT CERTAIN DATA TO MAKE PARTICULAR PARTS WORK
# READ AND PROCESS DATA
"""
#this is gonna take sooooooooooooooooooooooooooooooo long.
df1 = pd.read_csv("C:/Users/JZ/Desktop/Data mining/Project/accidents_2009_to_2011.csv", low_memory=False)
df2 = pd.read_csv("C:/Users/JZ/Desktop/Data mining/Project/accidents_2012_to_2014.csv", low_memory=False)
df3 = pd.read_csv("C:/Users/JZ/Desktop/Data mining/Project/ukTraffic.csv", low_memory=False)
df9 = pd.read_csv("C:/Users/JZ/Desktop/Data mining/Project/Vehicle_Information.csv", low_memory=False, encoding = 'unicode_escape')
df4 = pd.concat([df1, df2])
df = pd.merge(df4, df9, on='Accident_Index', how='left')
df = df.drop(columns=['Junction_Detail', 'Year_y', 'Junction_Control', 'Vehicle_Reference', 'Vehicle_Location.Restricted_Lane', 'Special_Conditions_at_Site'])

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)

# CLEAN DATA
df = df.drop_duplicates(keep='first')
df3 = df3.drop_duplicates(keep='first')
df = df.dropna()
df3 = df3.dropna()
df['Time']=pd.to_datetime(df['Time'], errors='coerce')
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df = df.rename(columns={"Did_Police_Officer_Attend_Scene_of_Accident": "Police_Attended", "Age_Band_of_Driver": "Driver_Age"})
df["Age_of_Vehicle"] = df["Age_of_Vehicle"].astype(int)
df['hour'] = df['Time'].dt.hour
df.fillna({'hour':1}, inplace=True)
df["hour"] = df["hour"].astype(int)
df = df[(df['hour']>=0) & (df['hour']<24)]
df = df[(df['Driver_Age']!='0 - 5') & (df['Driver_Age']!='Data missing or out of range') & (df['Driver_Age']!='6 - 10')]
df3 = df3.drop(columns=['Estimation_method', 'CP', 'LinkLength_km', 'LinkLength_miles', 'V2AxleRigidHGV', 'V3AxleRigidHGV', 'V4or5AxleRigidHGV', 'V3or4AxleArticHGV', 'V5AxleArticHGV', 'V6orMoreAxleArticHGV'])
df3 = df3[(df3['AADFYear']>=2009) & (df3['AADFYear']<=2014)]

#save data to make run time faster in the future
df.to_csv('C:/Users/JZ/Desktop/Data mining/accident_data.csv')
df3.to_csv('C:/Users/JZ/Desktop/Data mining/traffic_data.csv')
"""
#--------------------------------------------------------------------------------------------
#READ PROCESSED DATA 

df = pd.read_csv("C:/Users/JZ/Desktop/Data mining/accident_data.csv", low_memory=False)
df3 = pd.read_csv("C:/Users/JZ/Desktop/Data mining/traffic_data.csv", low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)
df['Time']=pd.to_datetime(df['Time'], errors='coerce')
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df = df.drop(columns=['Unnamed: 0'])
df3 = df3[(df3['Lon']>=-6.311496) & (df3['Lon']<=1.758722)]
df3 = df3[(df3['Lat']<=54.913811)]

#CREATE AND TRANSFORM VARIABLES

e = {'Dry': 1, 'Wet/Damp': 2, 'Frost/Ice':3, 'Snow':4, 'Flood (Over 3cm of water)':5}
df['Road_Surface'] = df['Road_Surface_Conditions'].map(e) 
f = {'One way street': 1, 'Single carriageway': 2, 'Dual carriageway':3, 'Roundabout':4, 'Slip road':5, 'Unknown':6}
df['Road__Type'] = df['Road_Type'].map(f) 
g = {'Going ahead other': 1, 'Wet/DampTurning right': 2, 'Reversing':3, 'Moving off':4,\
     'Overtaking moving vehicle - offside':5, 'Going ahead left-hand bend': 6, 'Overtaking static vehicle - offside': 7, \
     'Turning left': 8, 'Waiting to go - held up': 9, 'Slowing or stopping': 10, 'Parked': 11, \
     'U-turn': 12, 'Waiting to turn right': 12, 'Changing lane to right': 13, \
     'Overtaking - nearside': 14, 'Changing lane to left': 15, 'Waiting to turn left': 16, 'Data missing or out of range': 17}
df['Manoeuvre'] = df['Vehicle_Manoeuvre'].map(g) 
h = {'None': 1, 'Skidded': 2, 'Skidded and overturned':3, 'Overturned':4, 'Jackknifed':5, 'Jackknifed and overturned': 6, 'Data missing or out of range': 7}
df['Crash_Type'] = df['Skidding_and_Overturning'].map(h) 
dictionary = {1:"Fatal",
                2:"Serious",
                3:"Slight"}
df['severity'] = df['Accident_Severity'].map(dictionary) 
dictionary1 = {1:"Sun", 2:"Mon", 3:"Tue", 4:"Wed", 5:"Thu", 6:"Fri", 7:"Sat"}
df['dayofweek'] = df['Day_of_Week'].map(dictionary1)

#__________________________________________________________________________________________________________

# create yearly data of accident numbers and traffic congestion
traffic_sum = df3.groupby('AADFYear')['AllMotorVehicles'].sum().reset_index(name = "Traffic")
accident_sum = df['Date'].dt.year.value_counts().sort_index().reset_index(name = "Accidents")
accident_sum = accident_sum.rename(columns={'index': 'AADFYear'})
a = pd.merge(traffic_sum, accident_sum)

#plot that data 
b = a.set_index('AADFYear')['Traffic'].plot()
b.set_xlabel("YEAR")
b.set_ylabel("Traffic congestion")
plt.show()

b = a.set_index('AADFYear')['Accidents'].plot()
b.set_xlabel("YEAR")
b.set_ylabel("Number of accidents")
plt.show()
#____________________________________________________________________________________________________________

# create a heat map

plt.figure(figsize=(20,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#_________________________________________________________________________________________________________________

# REGRESSION

df = df.dropna() 
X = df[['Longitude', 'Latitude', 'Police_Force', 'Number_of_Vehicles',  'Day_of_Week', 'Local_Authority_(District)', \
        'Speed_limit',  'Urban_or_Rural_Area', 'Year_x', 'Accident_Severity',\
        'Age_of_Vehicle', 'hour', 'Road_Surface', 'Road__Type', 'Manoeuvre', 'Crash_Type']] #input
y = df['Number_of_Casualties'].values.reshape(-1,1) #output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = linear_model.LinearRegression() 
reg.fit(X_train, y_train) 

print('Intercept: \n', reg.intercept_)
print('Coefficients: \n', reg.coef_) 
print('Variance score: {}'.format(reg.score(X_train, y_train))) 

#-------------------------------------------------------------------------------------------

# CLUSTERING

# SELECT NECESSARY DATA, UNCOMMENT IF NECESSARY
df3 = df3[(df3['Lon']>=-0.3) & (df3['Lon']<=0.03)]
df3 = df3[(df3['Lat']>=51.4) & (df3['Lat']<=51.6)]
df = df[(df['Longitude']>=-0.3) & (df['Longitude']<=0.03)]
df = df[(df['Latitude']>=51.4) & (df['Latitude']<=51.6)]
df = df[(df['Accident_Severity']==1)]


#clustering accident data

x=df.loc[:, ['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=30)
kmeans = kmeans.fit(x)
labels = kmeans.labels_
df['cluster'] = labels
plt.scatter(x['Latitude'], x['Longitude'], s=50, c=labels, marker='.',\
            cmap=plt.cm.get_cmap('jet', 30))
plt.colorbar()
plt.show()
centers = kmeans.cluster_centers_
counts = np.bincount(labels)
result = np.where(counts == np.amax(counts))
result = result[0]
print(result)
most_accidents = centers[result]
f = result.item(0) # get the int of the cluster number
print(centers[f])
print(counts, f) 


#clustering traffic congestion data

x1=df3.loc[:, ['Lat', 'Lon']]
kmeans1 = KMeans(n_clusters=30)
kmeans1 = kmeans1.fit(x1)
labels1 = kmeans1.labels_
df3['cluster'] = labels1
plt.scatter(x1['Lat'], x1['Lon'], s=50, c=labels1, marker='.',\
            cmap=plt.cm.get_cmap('viridis', 30))
plt.colorbar()
plt.show()
centers1 = kmeans1.cluster_centers_
counts1 = np.bincount(labels1)
result1 = np.where(counts1 == np.amax(counts1))
result1 = result1[0]
print(result1)
most_accidents = centers1[result1]
f1 = result1.item(0)
print(centers1[f1])
print(counts1, f1)

#see cluster sizes
plt.figure(figsize=(8,5))
sns.countplot(x=df['cluster'], data=df)
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x=df3['cluster'], data=df3)
plt.show()

#-------------------------------------------------------------------------------------

#display folium map, works ONLY ON JUPITER notebook

location = df['Latitude'].mean(), df['Longitude'].mean()
m = folium.Map(location=location,zoom_start=6)

for i in range(0, len(centers)): #ACCIDENT DATA
    if(i==f):
        folium.CircleMarker(centers[i], radius=5, fill=True, color = 'red').add_to(m)
    else:
        folium.CircleMarker(centers[i], radius=5, fill=True, color = 'green').add_to(m)

for i in range(0, len(centers1)): #CONGESTION DATA
    if(i==f1):
        folium.CircleMarker(centers1[i], radius=5, fill=True, color = 'darkred').add_to(m)
    else:
        folium.CircleMarker(centers1[i], radius=5, fill=True, color = 'black').add_to(m)

#data_heat = df[['Latitude','Longitude']].values.tolist()
#plugins.HeatMap(data_heat).add_to(m)

m

#-------------------------------------------------------------------

plt.figure(figsize=(10,5))
df.groupby([df['Road_Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of accidents by road type')
plt.ylabel('Road type')
plt.xlabel('Number of cases')
plt.show() 


df = df[(df['Age_of_Vehicle']<=30)]
df = df[(df['Driver_Age']!='11 - 15')]

plt.figure(figsize=(10,5))
df.groupby([df['Driver_Age']]).size().sort_values(ascending=False).plot(kind='bar')
plt.title('Number of accidents by driver age')
plt.xlabel('Driver age')
plt.ylabel('Number of cases')
plt.xticks(
    rotation=0, 
    horizontalalignment='right')
plt.show() 


plt.figure(figsize=(10,5))
df.groupby([df['Age_of_Vehicle']]).size().sort_values(ascending=False).plot(kind='bar')
plt.title('Number of accidents by vehicle age')
plt.ylabel('Number of accidents')
plt.xlabel('Vehicle age')
plt.xticks(
    rotation=0, 
    horizontalalignment='right')
plt.show() 

#--------------------------------------------------------------------------------------

# WEATHER - not significant

plt.figure(figsize=(10,5))
df.groupby([df['Weather_Conditions']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of accidents by weather conditions')
plt.ylabel('Weather')
plt.xlabel('Number of cases')
plt.show() 

plt.figure(figsize=(10,5))
df.groupby([df['Road_Surface_Conditions']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of accidents by road hazards')
plt.ylabel('Road hazards')
plt.xlabel('Number of cases')
plt.show() 

#----------------------------------------------------------------------------------------

# SPEED LIMIT

df = df[(df['Accident_Severity']==1)]
#df = df[(df['Accident_Severity']!=1)]
# 1 is fatal, 3 is slight, use whichever is needed

a = df['Speed_limit'].value_counts().sort_index().reset_index(name = "Accidents")
a = a.rename(columns={'index': 'Limit'})
lines = a.plot.line(x='Limit', y='Accidents')
plt.show()

dfa = df[(df['Vehicle_Manoeuvre']!='Going ahead other')]
dfa.groupby(['Speed_limit', 'Vehicle_Manoeuvre']).size().unstack().plot(figsize=(10, 15), kind='bar', stacked=True)
plt.title('NUMBER OF ACCIDENTS BY SPEED LIMIT AND MANEUVRE')
plt.xlabel('SPEED LIMIT')
plt.ylabel('NUMBER OF ACCIDENTS')
plt.show()

plt.figure(figsize=(15,10))
dfa.groupby([dfa['Vehicle_Manoeuvre']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of accidents by vehicle maneuver')
plt.ylabel('Maneuver')
plt.xlabel('Number of accidents')
plt.show() 
#----------------------------------------------------------------------------------------

# PLOTS BY DAYS AND HOURS

#df = df[(df['Urban_or_Rural_Area']==1)]
#df = df[(df['Urban_or_Rural_Area']==2)]
# 1 for urban, 2 for rural, use whichever is needed

plt.figure(figsize=(10,5))
df.groupby([df['dayofweek']]).size().sort_values(ascending=False).plot(kind='bar')
plt.title('Number of accidents by day of the week')
plt.show() 

# GROUP HOURS INTO PARTS OF THE DAY
def when_was_it(hour):
    if hour >= 5 and hour < 8:
        return "early morning (5-7)"
    elif hour >= 7 and hour < 10:
        return "morning rush (7 - 9)"
    elif hour >= 9 and hour < 13:
        return "late morning (9 - 12)"
    elif hour >= 12 and hour < 16:
        return "early afternoon (12-15)"
    elif hour >= 15 and hour < 20:
        return "afternoon rush (15-19)"
    elif hour >= 19 and hour < 23:
        return "evening (19-22)"
    else:
        return "night (22-5)"
df['Day_Part'] = df['hour'].apply(when_was_it)

df.sort_values('dayofweek', ascending=False)
df.groupby(['dayofweek', 'Day_Part']).size().unstack().reindex(["Mon", "Tue", "Wed", 'Thu', 'Fri', 'Sat', 'Sun']).plot(figsize=(15,10), kind='bar', stacked=True, rot=0)
plt.title('NUMBER OF FATAL ACCIDENTS')
plt.ylabel('Number of cases')
plt.xlabel('Days of the week')
plt.show()

df.groupby(['hour', 'dayofweek']).size().unstack().reindex([0, 1, 2, 3, 4, 5, 6, 7, 8, \
           9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]).plot(figsize=(15,10), kind='line')
plt.title('Number of fatal accidents by day and hour')
plt.ylabel('Number of accidents')
plt.xlabel('Hour')
plt.show()

