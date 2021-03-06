# UK-Traffic-Accident-Analysis

For the purpose of this project, I will be using the UK traffic accidents datasets provided by the UK Department of Transport.

They can be found here: 

https://www.kaggle.com/daveianhickey/2000-16-traffic-flow-england-scotland-wales 

and

https://www.kaggle.com/tsiaras/uk-road-safety-accidents-and-vehicles

There are 2 datasets that contain all of the accident reports from 2009 to 2014, another dataset contains all vehicle and non-personal driver information for each of those accidents, as well as the UK traffic congestion data. The traffic congestion dataset provides a daily average count of vehicles that were on the UK roads. It does not have any common columns with the other datasets; however, it will be invaluable for providing insights into how changes in traffic congestion impact accidents. Before conducting data analysis and mining, it is crucial to merge these datasets properly, as well as perform data wrangling and preparation. As a result, there will be one dataset containing all necessary information and variables that will be needed to perform the analysis. 

The data cleaning and preparation performed on these datasets included dropping duplicate and missing information, converting data types into the ones that will be needed for analysis, as well as getting rid of impossible values that will skew the results of the analysis. Also, it was necessary to rename certain columns to make their contents more precise and short. In addition, new variables had to be created using dictionary maps in order to represent certain categorical variables as numerical in order to include them in the analysis. 

The main purpose of conducting this data analysis is to understand what can be done by both the UK government and local drivers to minimize the number of car accidents. In order to come up with the plan of action that would be taken, it is first necessary to discover factors that contribute to the accidents the most, including certain times, places, weather conditions, vehicles, or people. Finding this information will require utilizing variables that contain detailed per-accident information on each of those aspects. For instance, determining the most dangerous driving times would require variables such as date, time and day of the week. Using bar plots and grouping techniques can reveal that Fridays are generally the most dangerous days to drive because they see the highest overall number of accidents. Night driving is the most dangerous on Fridays and Saturdays, when more people would usually go out after dark. Surprisingly, the bar plots reveal that the morning rush hours see a lot less fatal accidents than non-fatal ones in comparison with the number of accidents in other parts of the day.  

A line graph showing hourly changes in overall number of accidents for every day of the week reveals that the most dangerous hours of the weekdays are morning and afternoon rush hours when people commute to and from work. On weekends, however, the most accidents happen around noon. If we restrict the data only to fatal accidents, however, we can see that the data is less consistent; however, Saturdays around 11AM and Monday afternoon rush hours seem to be the worst times to be on the roads. We can also see that weekend night hours experience as many fatal accidents as afternoon rush hours on other days.

However, in order to reveal more specific factors that contribute to the number and severity of the accidents, it is necessary to investigate the data further and see how variables could correlate or depend on each other. Creating a heat map could display these correlations clearly. In this case, it shows there are not a lot of strong correlations. Speed limit and urban or rural area seem to be strongly correlated, as well as variables that define accident and police locations. There are also correlations between the number of casualties and number of vehicles involved in accidents, between number of casualties and speed limit, between crash type and speed limit, urban or rural area, and road surface. 

In order to investigate if a number of casualties can be predicted by a number of factors such as number of vehicles involved in a crash, speed limits, accident severity and others, I ran a multiple linear regression analysis. The input variables were 'Longitude', 'Latitude', 'Police Force', 'Number of Vehicles',  'Day of Week', 'Local Authority (District)', 'Speed limit',  'Urban or Rural Area', 'Year_', ‘Accident Severity',  'Age of Vehicle', 'hour', 'Road Surface', 'Road Type', 'Maneuver', and 'Crash Type'. The output variable was the number of casualties. 

Intercept: [1.32276594]

Coefficients: 

 [[-2.84330452e-02  5.56134857e-02 -1.52708531e-05  3.66974789e-01
 
   1.44471544e-03  1.94156742e-05  5.14743346e-03  4.76178210e-02
   
  -1.66608120e-03 -2.01394819e-01  3.97791655e-03  6.98099207e-03
  
   1.13676563e-02 -2.52087193e-02 -7.67474490e-03  4.86835431e-02]]
   
Variance train score: 0.14783150972710124

These results indicate that for each additional road casualty, there are 3 more vehicles involved in an accident, an increase of speed limit by 5 points, and a significant increase in accident severity. Investigating the impact speed limits play in accidents further reveals that the majority of fatal accidents happen at speed of 60 and 30 mph. However, most non-fatal accidents happen mostly at the speed of 30 on single carriageways, making speed quite an impactful aspect.

Speed also plays an important role in maneuvers vehicles do at the time of the accidents. Aside from hitting an object or vehicle while going straight, the most dangerous road moves are turning right, overtaking a movie vehicle offside and being parked on the road. Turning right and changing lane to the right are both a lot more dangerous than turning left or changing the lane to the left. It is also dangerous to get held up while waiting to go.

Looking at the vehicle and driver information reveals some surprising information. One would expect older vehicles to be involved in road accidents more, possibly due to failed brakes, old tires that would skid in the rain, or other mechanical factors. However, plotting the data reveals that the majority of accidents involve newer vehicles, with 1-year old ones being the top. This can be explained by the fact that it may take time for people to get used to the new vehicles with all of the technology and features that could be very distracting. Also, the majority of the accidents are caused by people of ages between 25 and 45. Elderly people are involved in the least amount of accidents, indicating that people with the most driving experience are the safest drivers. Teen drivers possibly tend to be very careful on the roads due to their inexperience, which also can contribute to them having less accidents.

The last factor to be analyzed for impact on the UK road accidents is traffic congestion. In order to reveal how congestion relates to the number and location of accidents, k-means clustering analysis was used on both accidents and congestion datasets. Cluster centers, with the largest clusters marked in red, have been determined, plotted and put on an actual UK map. Visualizing this data indicated that especially in urban areas, congestion centers tend to be located geographically close to the accident hot spots, with the largest ones being in London downtown. In addition, clustering both datasets and putting both cluster centers on a map of London revealed that almost all major highways that connect downtown and suburban areas experience the most traffic jams and the majority of accidents around the same areas, confirming the relationship.

In conclusion, this data analysis has provided the information that could greatly help in reducing the UK road accidents. First, the government and local authorities can look into lowering the speed limits on single carriageways where the current limits are set at 30 miles per hour. Changing the speed limits to 55 or even 50 mph on large highways could also reduce the amount of road fatalities. Second, the government can look into highway expansion in order to reduce the congestion created by people traveling daily in and out of city centers. Local drivers, on the other hand, can pay extra attention while on single carriageways without separation between the directions because even a half-second distraction can end up in a head-on fatal collision. Drivers should also avoid speeding not only on highways but also on roads where the limits are lower (30mph) because this is actually where the majority of accidents happen. However, they should remember that driving slower than everyone else is also dangerous because people will be taking unnecessary risks and trying all possible ways to pass those slow drivers. As a result, there needs to be an effort from both the government and local drivers to reduce the amount of traffic accidents on UK roads.
