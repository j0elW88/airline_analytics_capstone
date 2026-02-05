from __future__ import annotations
import argparse
import math
from collections import defaultdict
import pandas as pd

ALIASES = {
    "year": ["Year", "YEAR"],
    "quarter": ["Quarter", "QUARTER"],
    "origin": ["Origin", "ORIGIN"],
    "dest": ["Dest", "DEST"],
    "mktID": ["MktID", "MKTID"],
    "ItinID": ["ItinID", "ITINID"],
    "origincountry": ["OriginCountry", "ORIGINCOUNTRY"],
    "airportgroup": ["AirportGroup", "AIRPORTGROUP"],
    "destcountry": ["DestCountry", "DESTCOUNTRY"],
    "carrier": ["TkCarrier", "OpCarrier", "CARRIER"],
    "tkcarrier": ["TkCarrier", "TKCARRIER"],
    "opcarrier": ["OpCarrier", "OPCARRIER"],
    "passengers": ["Passengers", "PASSENGERS", "PAX"],
    "deststatename": ["DestStateName", "DESTSTATENAME"],
    "nonstopmiles": ["NonStopMiles", "NONSTOPMILES"],
    "mktdistance": ["MktDistance", "MKTDISTANCE"],
    "mktmilesflown": ["MktMilesFlown", "MKTMILESFLOWN"],
    "fare": ["MktFare", "MARKET_FARE", "Fare"],
}
#########################################################################################################################

# OPTIONALLY SET PRICE SCALE TO REMOVE 
# CERTAIN OUTLIER PRICES FROM TESTING

#fare_upper_bound = 1200
#fare_lower_bound = 50

#########################################################################################################################




