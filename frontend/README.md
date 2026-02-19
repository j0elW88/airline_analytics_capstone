# Frontend

This frontend uses Streamlit and runs from "frontend/app.py".

## Setup (PowerShell)

powershell
cd "c:\Users\woest\Desktop\Soleus App\airline_analytics_capstone"
py -m pip install -r frontend\requirements.txt


## Run

```powershell
cd "c:\Users\woest\Desktop\Soleus App\airline_analytics_capstone"
py -m streamlit run frontend\app.py
```

## What this includes

- Home navigation (`History`, `Loaded Data Sets`, `Start`)
- Start flow (`Analyze One Period`, `Analyze Multiple Periods`, `Load Data Set`)
- Top-right `Back` button on all non-home screens
- Load by drag/drop or local path
- Auto-run `backend/capstone_parse.py` and `backend/capstone_analyze.py`
- Period completeness checks requiring all four files:
  - hub_market_power_<year>_Q<quarter>.csv`
  - route_market_power_<year>_Q<quarter>.csv`
  - hubxairline_<year>_Q<quarter>.csv`
  - routexairline_<year>_Q<quarter>.csv`



===================================================================================================================
TO DO

Beautify the front end, currently it is slop lol