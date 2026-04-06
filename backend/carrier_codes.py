from typing import Dict

CARRIER_LOOKUP: Dict[str, str] = {
    #d1db data set carrier codes and names (all)
    "3M": "Silver Airways",
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "C5": "CommutAir",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "G7": "GoJet Airlines",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "MX": "Breeze Airways",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "PT": "Piedmont Airlines",
    "QX": "Horizon Air",
    "SY": "Sun Country Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "XP": "Avelo Airlines",
    "YV": "Mesa Airlines",
    "YX": "Republic Airways",
    "ZW": "Air Wisconsin",
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
    #use as reference for mapping^^^
    if not isinstance(code, str):
        code = str(code)
    code = code.strip().upper()
    return CARRIER_LOOKUP.get(code, fallback if fallback is not None else code)
