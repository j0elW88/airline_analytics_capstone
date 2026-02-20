"""
Usage
from carrier_codes import get_carrier_name, CARRIER_LOOKUP
name = get_carrier_name("WN")     # "Southwest Airlines"
name = get_carrier_name("XX")     # "XX"  (graceful fallback)
"""

from typing import Dict

CARRIER_LOOKUP: Dict[str, str] = {
    #major US network carriers
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "US": "US Airways",              # merged into AA 2015; still in older data
    "NW": "Northwest Airlines",      # merged into DL 2010; in older data
    "CO": "Continental Airlines",    # merged into UA 2012; in older data
    "TW": "Trans World Airlines",    # merged into AA 2001; historical

    #low-cost carriers
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "SY": "Sun Country Airlines",
    "XP": "Avelo Airlines",
    "FL": "AirTran Airways",         # merged into WN 2014; in older data
    "HP": "America West Airlines",   # merged into US 2005; in older data
    "WA": "Western Airlines",        # merged into DL 1987; historical

    #regional carriers
    "OO": "SkyWest Airlines",
    "MQ": "Envoy Air",
    "YX": "Republic Airways",
    "9E": "Endeavor Air",
    "OH": "PSA Airlines",
    "PT": "Piedmont Airlines",
    "YV": "Mesa Airlines",
    "QX": "Horizon Air",
    "ZW": "Air Wisconsin",
    "CP": "Compass Airlines",
    "C5": "CommutAir",
    "RP": "Chautauqua Airlines",
    "EV": "ExpressJet Airlines",
    "XE": "ExpressJet Airlines",     # alternate code used for ExpressJet
    "GT": "Global Transpark Air",

    #alaska and hawaii
    "AS": "Alaska Airlines",
    "HA": "Hawaiian Airlines",
    "KH": "Aloha Air Cargo",

    #charter carriers
    "VX": "Virgin America",
    "G7": "GoJet Airlines",
    "3M": "Silver Airways",
    "SE": "Sun Country Airlines",
    "ZK": "Great Lakes Airlines",

    #cargo-only
    "5X": "UPS Airlines",
    "FX": "FedEx Express",

    #foreign carriers
    "WS": "WestJet",
    "AC": "Air Canada",
    "AM": "Aeromexico",

    #historical/defunct carriers
    "TZ": "ATA Airlines",
    "J7": "Midway Airlines",
    "XF": "Vanguard Airlines",
    "KP": "Kiwi International Air Lines",
    "PP": "Pacific Aviation",
    "AF": "Air Florida",
    "EA": "Eastern Airlines",
    "PA": "Pan American World Airways",
    "DH": "Independence Air",
    "TN": "Trans States Airlines",
    "TR": "Midwest Airlines",
}

#public helper
def get_carrier_name(code: str, fallback: str | None = None) -> str:
    """
    params
    code: str
        two-letter IATA / BTS TkCarrier code (e.g. "WN", "AA")
    fallback: str or None
        Value to return if the code is not in the lookup table
        Defaults to the code itself so the data is never blank
    returns
    str
        full airline name, or the fallback value
    example
    >>> get_carrier_name("WN")
    'Southwest Airlines'
    """
    if not isinstance(code, str):
        code = str(code)
    code = code.strip().upper()
    return CARRIER_LOOKUP.get(code, fallback if fallback is not None else code)
