Hello!

Welcome to the Blitz Analytics - Airline Analytics Capstone Project!

To Team:

Please work in your own branch to avoid conflicting or overwriting main!

To ALL: 

Before you begin we suggest you download any required Python Dependencies as well as the latest data from the Bureau of Transportation from the following link: 
https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK&QO_fu146_anzr=b4vtv0%20n0q%20Qr56v0n6v10%20f748rB 

Dependecy Requirements:
pandas >= 2.0.0
numpy >= 1.24.0
pyarrow >= 12.0.0

Please Ensure you Select the Following Fields (Some May Be Unnecessary We Just Suggest You Do These For Now):

* MKTID 
* YEAR
* QTR
* ORIGINAIRPORTID
* ORIGIN
* ORIGINSTATE
* ORIGINSTATENAME
* DESTAIRPORTID
* DEST
* DESTSTATE
* DESTSTATENAME
* AIRPORTGROUP
* WACGROUP
* TKCARRIER
* OPCARRIER
* BULKFARE
* PASSENGERS
* MKTFARE
* MKTMILESFLOWN
* NONSTOPMILES

Other Fields Are Optional But May Increase Your File Size!


Alternatively, navigate to the Bureau of Labor Statistics Website -> Go to Aviation -> and click the D1B1 Data -> D1B1 Market Data   (Name may differ slightly)

This file is approximately 1.6 GB so please allocate this space in your files adequately. 

Also note that this project has not been finalized, so excuse our dust!

We hope you find this tool helpful!

====================================================================================================================================================================================
To Those Whom This May Concern:

This following is meant to try to explain FAQ and the structure of this program.

Q. Whats with the CSVs?

Hub Airline File and Route Airline file's each are created for bug testing, these may not be necessary later, or may be saved to be utilized for comparisons later since theyre much smaller files.



Q. What are the data structures of this program?

Within the program, these are both saved as individual dictionaries of tuples.

Hub_airline and Route_airline use defaultdict(Agg)

hub_airline has the Tuple Key = (Origin, OriginState, Carrier)
route_airline has the Tuple Key = (Origin, Dest, Carrier)

These Tuple keys reference the Agg object which holds four values:
passengers_sum = summation of passengers
fare_x_passengers_sum = summation of (fare * passengers)
miles_x_passengers_sum = summation of (distance * passengers)
row_count = how many csv rows were counted to equal this #

This allows us to calculate everything else we need from the table



Q. How is file read? 

The file is read in chunks, currently set to 750,000 values at a time, and the dictionaries are updated each batch. This value can be increased or decreased (if you want to try to optimize this be my guest). 

In this time (during each batch) each chunk is turned into a "Dataframe" by pandas, then the "invalid" fares are dropped at this point, and the two helper columns (fare*passengers) and (distance*passengers) are calculated 



Q. So whats the output?

After all values are ingested, computations of avg_fare_weighted is calculated:
 (fare_x_passengers_sum / passengers_sum)
and same for avg_distance_weighted:
 (miles_x_passengers_sum / passengers_sum)
This data is then written to the CSV, and will likely be called upon soon to calculate HHI, expected markup, etc. 


Q. What do the 2 digit/letter codes mean under "Carrier"? / What does "99" code mean under carrier?

These are the airline carrier codes, we will need to remove the "99"s before our calculations, since these mean that the carrier was either "unknown" or was not assigned to a single airline. 


Q. Are layovers counted in this data?

Layovers are not counted in this data, instead we have opted to only count the flight from origin to dest, this may impact the avg distance weighted depending on the number of layovers typically taken, but this will be something we will discuss and figure out with our project advisor.

Q. What is the difference between Hub X Airline and Route X Airline

Hub measures an airlines prescence at a specific airport or "hub", regardless of the destination, therefore it measures all traffic that touches a given airport by a provider.
Route measures an airlines prescence on a specific city-pair, for example, how dominant is American Airlines on the ATL-MIA route. 
Therefore these two measure greatly different things, but are able to come from the same data set!S

========================================================================================================

Capstone_Analyze.py Functions: 
Responsible for market power analysis, reading in the csvs created by capstone_parser.py and creating two files, hub hhi & mkt pwr and route hhi & mkt pwr. This program then outputs hub_market_power_year_Q#.csv and route_market_power_year_Q#.csv. 

> Route-Level Calculations:

For each route let m = (origin,dest)
Let Q_im = passengers for airline i on route m
P_im = passenger-weighted fare for airline i on route m

Q_m_all = sum_k Q_km
(includes invalid carriers such as "99")

Q_m_valid = sum_{i in valid} Q_im
(only real airlines)

Market Share (includes valid airlines only):
share_im = Q_im / Q_m_valid

HHI:
HHI_m = (sum_{i in valid} share_im^2) × 10000

How to Interpret:

If HHI < 1500 : market is competitive 
If 1500 <= HHI <= 2500 : market is moderately concentrated
If HHI > 2500 : market is highly concentrated


Weighted Route Average Fare (includes invalid):
Pbar_m_all = (sum_k P_km × Q_km) / Q_m_all

Minimum Route Fare (includes invalid):
Pmin_m_all = min_k P_km

Markup Proxy (these are estimates not entirely accurate):
markup_im = P_im − Pbar_m_all

Lerner Proxy (these are estimates not entirely accurate):
lerner_proxy_im = (P_im − Pmin_m_all) / P_im

Note: 
Invalid carriers are used for baseline price estimation
Invalid carriers are NOT used in market share or HHI calculations

ROUTE Columns: 
Origin,Dest,Carrier,total_passengers,row_count,avg_fare_weighted,avg_distance_weighted,route_total_passengers_all,route_total_passengers_valid,carriers_on_route_all,carriers_on_route_valid,route_share,route_HHI,route_avg_fare_all,route_min_fare_all,markup_proxy_vs_route_avg,lerner_proxy_vs_route_min


> Hub Level Calculations: 

For each hub h = (Origin, OriginState):

Let:
Q_ih = total passengers for airline i at hub h

Q_h_valid = sum_{i in valid} Q_ih

Hub Share:
share_ih = Q_ih / Q_h_valid

Hub HHI:
HHI_h = (sum_{i in valid} share_ih²) × 10000

Hub HHI measures airline dominance at a specific airport.

This allows identification of:
Fortress hubs, Competitive airports, Dominant carriers

HUB Columns:
Origin,OriginState,Carrier,total_passengers,row_count,avg_fare_weighted,avg_distance_weighted,hub_total_passengers_all,hub_total_passengers_valid,carriers_at_hub_all,carriers_at_hub_valid,hub_share,hub_HHI,hub_avg_fare_all,hub_min_fare_all,markup_proxy_vs_hub_avg,lerner_proxy_vs_hub_min

========================================================================================================

Things to be done still:


> Build the analysis runner (prints your required metrics)

Create analysis_report.py that reads those two CSVs and prints:

Overall average fare (passenger-weighted) of carriers 
Market size (total passengers; passenger-miles)
Top hubs by passengers Top routes by passengers (demand) 
Highest-cost routes (min passenger cutoff)
Airline avg fares + lowest/highest airline
Route density (passenger-miles)
Revenue proxy (passengers × fare)

Add HHI + market shares:
Route-level HHI from routexairline
Hub-level HHI from hubxairline


> Add “markup” + price discrimination (properly framed)

Choose markup baseline and implement:
Route-mean markup proxy, or Predicted-fare markup proxy (fare ~ distance)

Price discrimination proxies:
fare vs HHI fare-per-mile vs distance bucket hub premium vs non-hub


>  Multi-period comparisons (seasonality + growth)

Run parser for multiple quarters/years and keep outputs.

Build a combiner that stacks periods into:
hub_all_periods.csv
route_all_periods.csv

Compute growth/decline rates by:
carrier, hub, route, region

 
