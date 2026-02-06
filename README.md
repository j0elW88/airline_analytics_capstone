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


