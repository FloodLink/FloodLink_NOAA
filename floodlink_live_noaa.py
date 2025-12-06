"""
FloodLink – Live Flood Risk Evaluator (RAW + Linear)
Evaluates high-risk features from Citiesglobal.csv using NOAA GFS forecasts.

Now includes:
- Configurable forecast horizon (3h, 6h, 12h, etc.)
- Linear, unit-aware multipliers (rain unbounded; soil & RH clipped)
- RAW score only (no compression)
- Level-transition alerts only (Medium↔High, High↔Extreme; downgrades toggle)
- Single-file comparison (alerts_comparison.json)
- Rich Tweet Tracker (tweeted_alerts.json)
"""

import os
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tweepy
from requests.exceptions import RequestException, ReadTimeout, ConnectionError

import numpy as np
from scipy.interpolate import griddata
import pygrib

# -------------------------------
# CONFIGURATION
# -------------------------------
CSV_PATH = "cities15000.csv"
COMPARISON_PATH = "alerts_comparison.json"   # single source of truth
TWEET_LOG_PATH = "tweeted_alerts.json"       # map-ready tweet history

SLEEP_BETWEEN_CALLS = 0.1         # seconds between API calls (not used for bulk)
COMPARISON_HISTORY = 5  # or 10
TIMEZONE = "Europe/Madrid"
MAX_RETRIES = 1                    # Increased for downloads
TIMEOUT = 30                       # request timeout (s) per NOAA download
FORECAST_HOURS = 6                # Extended to 12 for NOAA

# --- Twitter config ---
TWITTER_ENABLED = os.getenv("TWITTER_ENABLED", "false").lower() == "true"
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_SECRET = os.getenv("TWITTER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
MIN_SECONDS_BETWEEN_TWEETS = 30

# -------------------------------
# TUNABLE CONSTANTS (units!)
# -------------------------------
RISK_THRESHOLD = 8.5         # baseline FRisk cutoff from GIS layer

RAIN_UNIT_MM   = 100.0       # 100 mm → 1.0× rain multiplier
SOIL_MIN_MULT  = 0.95         # soil=0 -> 0.8×
SOIL_MAX_MULT  = 1.8         # soil=1 -> 1.8×
HUM_MIN_MULT   = 1.0         # RH=0% -> 0.9×
HUM_MAX_MULT   = 1.05         # RH=100% -> 1.1×
RAIN_CUTOFF_MM = 0.0         # set 0.5 to ignore drizzle; 0.0 keeps strict linearity

# RAW alert bands (tune later or learn from rolling percentiles)
RAW_LOW_MAX   = 5.0          # 0..5   -> Low
RAW_MED_MAX   = 15.0         # 5..15  -> Medium
RAW_HIGH_MAX  = 35.0         # 15..35 -> High
# >35 -> Extreme

# -------------------------------
# ALERT TRANSITION POLICY
# -------------------------------
TWEET_LEVELS = ["Medium", "High", "Extreme"]   # which levels are tweet-worthy at all
ALERT_ON_UPGRADES   = True                     # Medium→High, High→Extreme
ALERT_ON_DOWNGRADES = True                     # High→Medium, Extreme→High

LEVELS = ["None", "Low", "Medium", "High", "Extreme"]

# -------------------------------
# NOAA GFS CONFIG
# -------------------------------
GFS_RES = '0p50'  # Use 0.5° for smaller files (<10 MB); change to '0p25' for finer res
VARIABLES = ['APCP', 'RH', 'SOILW']
LEVELS_DICT = {
    'APCP': 'surface',
    'RH': '2 m above ground',
    'SOILW': '0-0.1 m below ground'  # Corrected for GFS standard
}

def get_latest_cycle():
    now = datetime.utcnow()
    date = now.strftime('%Y%m%d')
    cycle_hour = (now.hour // 6) * 6
    cycle = f"{cycle_hour:02d}"
    # Fallback if latest not ready: subtract 6h
    if cycle_hour == 0:  # Edge case midnight
        prev_date = (now - timedelta(days=1)).strftime('%Y%m%d')
        prev_cycle = '18'
        return date, cycle, prev_date, prev_cycle
    prev_cycle = f"{cycle_hour - 6:02d}"
    return date, cycle, date, prev_cycle

def download_gfs_file(date, cycle, fhr):
    base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{GFS_RES}.pl"
    grid_type = "pgrb2full" if GFS_RES == '0p50' else "pgrb2"
    params = {
        "file": f"gfs.t{cycle}z.{grid_type}.{GFS_RES}.f{fhr:03d}",
        "dir": f"/gfs.{date}/{cycle}/atmos"
    }
    for var in VARIABLES:
        params[f"var_{var}"] = "on"
        params[f"lev_{LEVELS_DICT[var]}"] = "on"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(base_url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            file_path = f"gfs_{cycle}_f{fhr:03d}.grb2"
            with open(file_path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {file_path} (size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB)")
            return file_path
        except Exception as e:
            print(f"Download failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"URL tried: {r.url if 'r' in locals() else base_url + '?' + '&'.join([k+'='+v for k,v in params.items()])}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def load_gfs_grids(forecast_hours):
    date, cycle, prev_date, prev_cycle = get_latest_cycle()
    grids = {var: [] for var in VARIABLES}
    times = []
    lats, lons = None, None
    for fhr in range(1, forecast_hours + 1):  # f001 to f012 for next 12h
        file = download_gfs_file(date, cycle, fhr)
        if file is None:
            # Fallback to previous cycle
            file = download_gfs_file(prev_date, prev_cycle, fhr)
            if file is None:
                continue
        grb = pygrib.open(file)
        for var in VARIABLES:
            if var == 'SOILW':
                msg = grb.select(name='Volumetric soil moisture content', typeOfLevel='depthBelowLandLayer', bottomLevel=0.1, topLevel=0)[0]
            elif var == 'RH':
                msg = grb.select(name='Relative humidity', typeOfLevel='heightAboveGround', level=2)[0]
            else:  # APCP
                msg = grb.select(name='Total Precipitation')[0]
            grids[var].append(msg.values)
            if lats is None:
                lats, lons = msg.latlons()
        times.append(msg.validDate)  # UTC
        grb.close()
        os.remove(file)
    for var in grids:
        grids[var] = np.stack(grids[var]) if grids[var] else None
    return grids, lats, lons, times

# -------------------------------
# WEATHER INDICATORS
# -------------------------------
def compute_indicators(grids, lats, lons, times, lat, lon):
    """
    Use the next FORECAST_HOURS from GFS grids.

    Returns:
        rain_sum (mm),
        rh_avg (%),
        soil_avg (0–1),
        peak_dt_local (datetime or None)  # local time of max rainfall in window
    """
    if any(g is None for g in grids.values()):
        return 0.0, 0.0, 0.0, None
    points = np.column_stack((lats.ravel(), lons.ravel()))
    rain_vals = []
    rh_vals = []
    soil_vals = []
    for t in range(min(len(times), FORECAST_HOURS)):
        # For precip: Incremental (diff from previous; assume f000=0 or handle)
        if t == 0:
            rain_inc = griddata(points, grids['APCP'][t].ravel(), (lat, lon), method='nearest')
        else:
            rain_inc = griddata(points, grids['APCP'][t].ravel(), (lat, lon), method='nearest') - griddata(points, grids['APCP'][t-1].ravel(), (lat, lon), method='nearest')
        rain_vals.append(max(0, rain_inc))  # Avoid negative artifacts
        rh_vals.append(griddata(points, grids['RH'][t].ravel(), (lat, lon), method='nearest'))
        soil_vals.append(griddata(points, grids['SOILW'][t].ravel(), (lat, lon), method='nearest'))
    rain_sum = sum(rain_vals)
    rh_avg = sum(rh_vals) / len(rh_vals) if rh_vals else 0.0
    soil_norm = [min(max(x / 0.6, 0.0), 1.0) for x in soil_vals]  # Normalize like Open-Meteo (GFS max ~0.6)
    soil_avg = sum(soil_norm) / len(soil_norm) if soil_norm else 0.0
    if any(rain_vals):
        max_idx = np.argmax(rain_vals)
        peak_dt_utc = times[max_idx].replace(tzinfo=ZoneInfo('UTC'))
        # Convert to local timezone (approximate; use location-specific if needed)
        tz = ZoneInfo(TIMEZONE)  # Fallback; ideally use per-location tz
        peak_dt_local = peak_dt_utc.astimezone(tz)
    else:
        peak_dt_local = None
    return rain_sum, rh_avg, soil_avg, peak_dt_local

# -------------------------------
# LINEAR MULTIPLIERS
# -------------------------------
def rainfall_multiplier(rain_mm: float) -> float:
    return max(0.0, rain_mm / RAIN_UNIT_MM)

def soil_multiplier(soil_frac: float) -> float:
    s = max(0.0, min(1.0, soil_frac))
    return SOIL_MIN_MULT + s * (SOIL_MAX_MULT - SOIL_MIN_MULT)

def humidity_multiplier(rh_percent: float) -> float:
    rh = max(0.0, min(100.0, rh_percent))
    return HUM_MIN_MULT + (rh / 100.0) * (HUM_MAX_MULT - HUM_MIN_MULT)


# -------------------------------
# RISK MODEL (RAW ONLY)
# -------------------------------
def calculate_dynamic_risk_raw(base_risk: float, rain_mm: float, rh_percent: float, soil_frac: float):
    """
    Returns: (raw_score, level, r_mult, s_mult, h_mult)
    raw_score is linear in rain, soil, humidity (multiplicative across factors).
    """
    if rain_mm < RAIN_CUTOFF_MM:
        return 0.0, "None", 0.0, soil_multiplier(0.0), humidity_multiplier(0.0)

    r_mult = rainfall_multiplier(rain_mm)
    s_mult = soil_multiplier(soil_frac)
    h_mult = humidity_multiplier(rh_percent)

    raw_score = max(0.0, base_risk) * r_mult * s_mult * h_mult

    if raw_score == 0:
        level = "None"
    elif raw_score < RAW_LOW_MAX:
        level = "Low"
    elif raw_score < RAW_MED_MAX:
        level = "Medium"
    elif raw_score < RAW_HIGH_MAX:
        level = "High"
    else:
        level = "Extreme"

    return round(raw_score, 3), level, r_mult, s_mult, h_mult


# -------------------------------
# ALERT COMPARISON (level transitions only)
# -------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": []}

def rotate_comparison_snapshots(max_history=COMPARISON_HISTORY):
    """
    Rotate alerts_comparison snapshots:

      alerts_comparison_{max_history-1}.json -> alerts_comparison_{max_history}.json
      ...
      alerts_comparison_1.json -> alerts_comparison_2.json
      alerts_comparison.json   -> alerts_comparison_1.json

    The new current run will then be written to alerts_comparison.json.
    """
    base = COMPARISON_PATH  # "alerts_comparison.json"

    # Shift numbered snapshots up: N-1 -> N, ..., 1 -> 2
    for i in range(max_history - 1, 0, -1):
        older = f"alerts_comparison_{i}.json"
        newer = f"alerts_comparison_{i + 1}.json"
        if os.path.exists(older):
            if os.path.exists(newer):
                os.remove(newer)
            os.replace(older, newer)

    # Move current base file to _1
    if os.path.exists(base):
        first_snapshot = "alerts_comparison_1.json"
        if os.path.exists(first_snapshot):
            os.remove(first_snapshot)
        os.replace(base, first_snapshot)


def build_alert_dict(alerts):
    return {(round(a["latitude"], 4), round(a["longitude"], 4)): a for a in alerts}

def compare_alerts(prev, curr):
    """
    Tweet when:
      • First time we see a site at a tweet-worthy level (Medium/High/Extreme)
      • Any UPGRADE into a tweet-worthy level (e.g., None→Medium, Low→Medium, Medium→High, High→Extreme)
      • (Optional) Downgrades if enabled
    """
    changes = []
    for key, c in curr.items():
        cur_lvl = c["dynamic_level"]

        # New site this run
        if key not in prev:
            if cur_lvl in TWEET_LEVELS:
                changes.append(("New", c))
            continue

        prev_lvl = prev[key]["dynamic_level"]
        if prev_lvl == cur_lvl:
            continue

        prev_i, cur_i = LEVELS.index(prev_lvl), LEVELS.index(cur_lvl)

        # Any upgrade into a tweet-worthy level
        if ALERT_ON_UPGRADES and cur_i > prev_i and cur_lvl in TWEET_LEVELS:
            changes.append(("Upgrade", c))
            continue

        # Downgrades from tweet-worthy levels (optional)
        if ALERT_ON_DOWNGRADES and cur_i < prev_i and prev_lvl in TWEET_LEVELS:
            changes.append(("Downgrade", c))

    return changes


# -------------------------------
# TWEET MANAGEMENT
# -------------------------------
def load_tweeted_alerts():
    if os.path.exists(TWEET_LOG_PATH):
        with open(TWEET_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_tweeted_alerts(tweeted):
    with open(TWEET_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(tweeted, f, indent=2, ensure_ascii=False)

def cleanup_tweeted_alerts(tweeted, valid_coords):
    """
    Keep only:
      - coordinates that still exist in the CSV, AND
      - entries NOT marked as resolved.

    Entries are marked resolved=True when a downgrade to Low/None happens.
    They stay in tweeted_alerts.json for that run, and are removed here on
    the following run.
    """
    cleaned = {}
    for k, v in tweeted.items():
        # k is "lat,lon" string
        if k not in valid_coords:
            continue
        # Drop entries that have already been marked as resolved
        if v.get("resolved", False):
            continue
        cleaned[k] = v

    if len(cleaned) < len(tweeted):
        print(f"🧹 Cleaned {len(tweeted) - len(cleaned)} outdated tweet entries.")
    return cleaned

def tweet_alert(change_type, alert, quote_tweet_id=None):
    """Post a tweet for a new or transitioned flood alert."""
    lat, lon = alert["latitude"], alert["longitude"]
    level = alert["dynamic_level"]

    # 🎨 Emoji color map for risk level
    level_colors = {
        "None": "⚪",
        "Low": "⚪",
        "Medium": "🟢",
        "High": "🟠",
        "Extreme": "🔴",
    }

    # Get the appropriate color or default to ⚪
    color_emoji = level_colors.get(level, "⚪")

    # Place string like "Kuantan, Malaysia"
    place = ", ".join(
        [x for x in [alert.get("name", "Location"), alert.get("country", "")] if x]
    )

    # Uppercase level for the header line
    level_upper = level.upper()
    
    peak_time_str = alert.get("peak_time_local_str", "unknown")

    tweet_text = (
        f"{color_emoji} {level_upper} FLOOD RISK – {place}\n\n"
        f"Type: {change_type}\n"
        f"Local Time: {peak_time_str}\n"
        f"Location: ({lat:.2f}, {lon:.2f})\n"
        f"Rain: {alert[f'rain_{FORECAST_HOURS}h_mm']:.1f} mm\n"
        # f"Soil moisture: {alert['soil_moisture_avg']:.2f}\n"
        # f"Humidity: {alert['humidity_avg']}%\n"
    )

    print(f"🚨 Tweet → {tweet_text}\n" + (f"(Quoting ID: {quote_tweet_id})\n" if quote_tweet_id else ""))

    if not TWITTER_ENABLED:
        print("🧪 DRY RUN (tweet suppressed). Set TWITTER_ENABLED=true to send.")
        return None  # No ID in dry run

    try:
        client = tweepy.Client(
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            wait_on_rate_limit=True,
        )
        response = client.create_tweet(
            text=tweet_text,
            quote_tweet_id=quote_tweet_id  # None is fine, ignored if absent
        )
        new_tweet_id = response.data['id']
        print(f"✅ Tweet posted with ID: {new_tweet_id}")
        return str(new_tweet_id)  # Return as str for JSON safety
    except Exception as e:
        print(f"❌ Tweet failed: {e}")
        return None


# -------------------------------
# MAIN WORKFLOW
# -------------------------------
def main():
    print(f"🌧️ FloodLink Live Risk Evaluation started ({FORECAST_HOURS}-hour window)…")

    previous = load_json(COMPARISON_PATH)
    prev_alerts_dict = build_alert_dict(previous.get("alerts", []))
    tweeted_alerts = load_tweeted_alerts()

    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH} – skipping evaluation.")
        return  # Or handle as needed

    df = pd.read_csv(CSV_PATH)
    high_risk = df[df["FRisk"] > RISK_THRESHOLD].copy()

    valid_coords = {f"{row['Latitude']:.4f},{row['Longitude']:.4f}" for _, row in df.iterrows()}
    tweeted_alerts = cleanup_tweeted_alerts(tweeted_alerts, valid_coords)

    alerts = []
    start_time = time.time()

    # Load NOAA GFS grids once for all locations
    grids, lats, lons, times = load_gfs_grids(FORECAST_HOURS)
    if grids is None or any(g is None for g in grids.values()):
        print("❌ Failed to load GFS data – using previous alerts where available.")
        # Fallback to previous for all
        for _, row in high_risk.iterrows():
            key = (round(row["Latitude"], 4), round(row["Longitude"], 4))
            prev_alert = prev_alerts_dict.get(key)
            if prev_alert:
                alerts.append(prev_alert)
            # Skip new ones
        # Proceed with what we have
    else:
        for _, row in high_risk.iterrows():
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
            base_risk = float(row["FRisk"])
            name = str(row.get("ETIQUETA", f"id_{row['JOIN_ID']}"))
            country = str(row.get("Country", "")).strip()

            rain_sum, rh_avg, soil_avg, peak_dt_local = compute_indicators(grids, lats, lons, times, lat, lon)

            raw_score, dyn_level, r_mult, s_mult, h_mult = calculate_dynamic_risk_raw(
                base_risk, rain_sum, rh_avg, soil_avg
            )

            # Simple local time string like "23:00"
            if peak_dt_local is not None:
                peak_time_local_str = peak_dt_local.strftime("%H:%M")
            else:
                peak_time_local_str = "unknown"

            alerts.append({
                "id": str(row["JOIN_ID"]),
                "country": country,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "base_risk": round(base_risk, 2),

                f"rain_{FORECAST_HOURS}h_mm": round(rain_sum, 2),
                "humidity_avg": round(rh_avg, 1),
                "soil_moisture_avg": round(soil_avg, 3),

                # Diagnostics for tuning
                "rain_mult": round(r_mult, 3),
                "soil_mult": round(s_mult, 3),
                "humidity_mult": round(h_mult, 3),

                "raw_dynamic_score": raw_score,
                "dynamic_level": dyn_level,

                # NEW: time of max rainfall in the window
                "peak_time_local_str": peak_time_local_str,
            })

    # Persist current results
    result = {
        "timestamp": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "source": "NOAA GFS",
        "forecast_window_hours": FORECAST_HOURS,
        "features_evaluated": len(alerts),
        "alerts": alerts
    }

    # Detect level-change events
    curr_alerts_dict = build_alert_dict(alerts)
    changes = compare_alerts(prev_alerts_dict, curr_alerts_dict)
    print(f"🔍 Detected {len(changes)} level-change events.")

    # 👉 Debug: list each transition with prev → current (plus key metrics)
    if changes:
        for change_type, a in changes:
            key = (round(a["latitude"], 4), round(a["longitude"], 4))
            prev_lvl = prev_alerts_dict.get(key, {}).get("dynamic_level", "None")
            print(
                "🛰️ "
                f"{a['name']} [{a['latitude']:.4f},{a['longitude']:.4f}]: "
                f"{prev_lvl} → {a['dynamic_level']} ({change_type}); "
                f"rain={a[f'rain_{FORECAST_HOURS}h_mm']} mm, "
                f"soil={a['soil_moisture_avg']:.3f}, RH={a['humidity_avg']}%"
            )
    else:
        print("ℹ️ No tweetable transitions this run (either steady level or below tweet-worthy).")

    last_tweet_ts = 0.0

    # Tweet + update tracker
    for change_type, alert in changes:
        key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
        current_level = alert["dynamic_level"]
        last_entry = tweeted_alerts.get(key)

        # --- Downgrade gating logic ---
        if change_type == "Downgrade":
            # If we've never tweeted this location, ignore the downgrade
            if last_entry is None:
                print(
                    f"↘️ Skipping downgrade tweet for {key} "
                    f"({alert['name']}) – no prior tweet recorded."
                )
                continue

            # If the last tweeted level is already Low/None (i.e. not in TWEET_LEVELS),
            # we've already announced the downgrade for this alert cycle.
            last_level = last_entry.get("risk_level", "None")
            if last_level not in TWEET_LEVELS:
                print(
                    f"↘️ Skipping extra downgrade tweet for {key} "
                    f"({alert['name']}) – last tweeted level is already "
                    f"{last_level} (outside {TWEET_LEVELS})."
                )
                continue

        # -------------------------
        # Stream-wide rate limiting
        # -------------------------
        now_ts = time.time()
        if now_ts - last_tweet_ts < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts))

        # --- Quote logic: Use previous tweet ID if available (for upgrades/downgrades) ---
        quote_tweet_id = None
        if change_type in ["Upgrade", "Downgrade"] and last_entry and "tweet_id" in last_entry:
            quote_tweet_id = last_entry["tweet_id"]

        # Send tweet (or DRY RUN printout), capture new ID
        new_tweet_id = tweet_alert(change_type, alert, quote_tweet_id=quote_tweet_id)
        last_tweet_ts = time.time()

        # --- Update tweeted_alerts.json according to the new level ---
        if new_tweet_id:  # Only update if tweet succeeded
            if current_level in TWEET_LEVELS:
                # Still Medium / High / Extreme → keep or create/update entry
                tweeted_alerts[key] = {
                    "country": alert.get("country", ""),
                    "name": alert["name"],
                    "risk_level": current_level,
                    "latitude": alert["latitude"],
                    "longitude": alert["longitude"],
                    "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                    "humidity": alert["humidity_avg"],
                    "soil_moisture": alert["soil_moisture_avg"],
                    "raw_dynamic_score": alert["raw_dynamic_score"],
                    "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
                    "tweet_id": new_tweet_id  # NEW: Store/update the latest tweet ID
                }
            else:
                # Downgrade into Low / None → keep it ONE more run as 'resolved'
                print(
                    f"✅ Marking alert as resolved in tweet log: "
                    f"{alert['name']} [{key}] (→ {current_level})"
                )

                tweeted_alerts[key] = {
                    "country": alert.get("country", ""),
                    "name": alert["name"],
                    "risk_level": current_level,  # "Low" or "None"
                    "latitude": alert["latitude"],
                    "longitude": alert["longitude"],
                    "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                    "humidity": alert["humidity_avg"],
                    "soil_moisture": alert["soil_moisture_avg"],
                    "raw_dynamic_score": alert["raw_dynamic_score"],
                    "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
                    "resolved": True,  # <-- flag for next run's cleanup
                    "tweet_id": new_tweet_id  # NEW: Store the downgrade tweet ID (optional, but keeps chain info)
                }

    save_tweeted_alerts(tweeted_alerts)

    # Rotate old comparison snapshots, then write the new one
    rotate_comparison_snapshots(COMPARISON_HISTORY)

    # Update comparison file
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed in {round((time.time() - start_time)/60, 1)} min. "
          f"Updated {COMPARISON_PATH} and {TWEET_LOG_PATH}.")

# -------------------------------
if __name__ == "__main__":
    main()
