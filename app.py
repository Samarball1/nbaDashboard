from flask import Flask, render_template, request
from nba_api.stats.static import teams
import pandas as pd
import requests
import json
import time
from datetime import date, datetime
import threading


def get_nba_teams():
    nba_teams = teams.get_teams()
    return sorted([(team["full_name"], team["abbreviation"]) for team in nba_teams])


def pull_unique_vals():
    url = 'https://stats.nba.com/stats/leaguedashplayerstats'
    cache_file = "nba_players_cache.json"

    # Default players in case of failure
    default_players = [
        ("LeBron James", 2544),
        ("Giannis Antetokounmpo", 203507),
        ("Luka Dončić", 1629029),
    ]

    # Check if cached data exists
    try:
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
            if cached_data:
                return sorted([(player["PLAYER_NAME"], player["PLAYER_ID"]) for player in cached_data])
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Proceed with API call if cache is missing or corrupted

    params = {
        {
  "MeasureType": "Base",
  "PerMode": "Totals",
  "PlusMinus": "N",
  "PaceAdjust": "N",
  "Rank": "N",
  "LeagueID": "00",
  "Season": "2024-25",
  "SeasonType": "Regular Season",
  "PORound": 0,
  "Outcome": null,
  "Location": null,
  "Month": 0,
  "SeasonSegment": null,
  "DateFrom": null,
  "DateTo": null,
  "OpponentTeamID": 0,
  "VsConference": null,
  "VsDivision": null,
  "TeamID": 0,
  "Conference": null,
  "Division": null,
  "GameSegment": null,
  "Period": 0,
  "ShotClockRange": null,
  "LastNGames": 0,
  "GameScope": null,
  "PlayerExperience": null,
  "PlayerPosition": null,
  "StarterBench": null,
  "DraftYear": null,
  "DraftPick": null,
  "College": null,
  "Country": null,
  "Height": null,
  "Weight": null,
  "TwoWay": null,
  "GameSubtype": null,
  "ActiveRoster": null,
  "ISTRound": null
}
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Host": "stats.nba.com",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

    # Retry mechanism in case of failures
    for attempt in range(3):
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error if request fails

        data = response.json()
        first = data['resultSets'][0]
        headers = first['headers']
        rows = first['rowSet']

        if not rows:
            print("No player data found.")
            return default_players  # Return default players on empty response

        df_stats = pd.DataFrame(rows, columns=headers)
        df_stats = df_stats.sort_values('PTS_RANK', ascending=False, inplace = True).reset_index()
        print("check: player values: ", df_stats)
        players = [(row['PLAYER_NAME']) for row in df_stats.iterrows()]

        return df_stats


# output = widgets.Output()

# Function to fetch and display NBA game logs
fetched = False
def pull_nba_api_dates(start, end, season):

    # clear_output(wait=True)  # Clear previous output before showing new data
    print(f"Fetching NBA game logs from {start} to {end}... for")  # Show loading message
    
    url = "https://stats.nba.com/stats/leaguegamelog"
    
    params = {
        "Counter": 1000,
        "DateFrom": start,
        "DateTo": end,
        "Direction": "DESC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "DATE",
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nba.com",
    }
    results = 0

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raise error for failed requests

    data = response.json()
    first = data['resultSets'][0]
    headers = first['headers']
    rows = first['rowSet']

    if not rows:
        print("No games found for this date range.")
        return
    
    results = pd.DataFrame(rows, columns=headers)
    #filtering based on params
        
    return results
            
    print(f"Fetching season {season}...")

# ✅ Fixed List of Seasons
seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
# seasons = ["2024-25"]


def pull_nba_api_all_seasons(start, end):
    """
    Loops through a fixed list of seasons and concatenates all results.
    Uses `pull_nba_api_dates(start, end)` without additional API checks.
    """
    all_results = []  # List to store results

    for season in seasons:
        print(f"Fetching season {season}...")
        
        # Modify `params` inside `pull_nba_api_dates` to update season
        results = pull_nba_api_dates(start, end, season)  # Calls your function
        
        if results is not None:
            results["Season"] = season  # Add season column
            all_results.append(results)  # Store data
            
        time.sleep(1)  # Prevent potential rate limit issues

    # ✅ Concatenate all results into one DataFrame
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        fetched = True
        print("NBA FETCH COMPLETE")
        return final_results
    else:
        print("No data found for any season.")
        return None

# ✅ Example Usage
# Display the combined DataFrame 
main_stats = ["PTS", "REB", "AST", "P+R+A", "STL", "BLK", "TOV", "PLUS_MINUS", "MIN", "FG_PCT", "FG3_PCT"]

def get_avg_stats(r, title):
        numeric_results = r[main_stats].apply(pd.to_numeric, errors="coerce")

        # ✅ Compute averages & Properly Round
        avg_stats1 = numeric_results.mean().round(2).to_frame().T  # Convert to DataFrame

        # ✅ Convert to string to remove float64 issue
        avg_stats1 = avg_stats1.astype(str)

        # ✅ Format percentage columns (FG_PCT, FG3_PCT)
        percent_cols = ["FG_PCT", "FG3_PCT"]
        for col in percent_cols:
            if col in avg_stats1.columns:
                avg_stats1[col] = avg_stats1[col].apply(lambda x: f"{float(x)*100:.1f}%" if x != 'nan' else "N/A")

        # ✅ Add title column to the stats
        # avg_stats1.insert(0, "Stat Type  ", f"Average Stats {title}: ")
        avg_stats1= avg_stats1.rename({'PLUS_MINUS': '+ -' }, axis=1)

        return avg_stats1

 

def func_filter_nba(start, end, periods = None, player=None, opponent=None, all_seasons_data=None, players = None, teams = None): 
    if not player:
        print("PLEASE SELECT PLAYER FIRST")
        return
    results = all_seasons_data
    results['GAME_DATE'] = pd.to_datetime(results['GAME_DATE'])
    results['P+R+A'] = results['PTS'] + results['AST'] + results['REB']


    # Extract Opponent abbreviation
    results["Opponent"] = results["MATCHUP"].str.extract(r"(?:vs\.|@)\s([A-Z]{3})")

    # ✅ Filter by date range
    results = results[(results['GAME_DATE'] >= pd.Timestamp(start)) & (results['GAME_DATE'] <= pd.Timestamp(end))]
        
    resultsd = results.copy()

    
    # ✅ Filter by player
    players = pd.DataFrame(players, columns=["PLAYER_NAME", "PLAYER_ID"])

    if player:
        play_cond = players[players['PLAYER_NAME'] == player]['PLAYER_ID']
        if not play_cond.empty:
            results = results[results['PLAYER_ID'] == play_cond.iloc[0]]
            resultsd = results[results['PLAYER_ID'] == play_cond.iloc[0]]

    # ✅ Filter by opponent
    teams = pd.DataFrame(teams, columns=["Team", "Abbreviation"])

    if opponent:
        abv_cond = teams[teams['Team'] == opponent]['Abbreviation']
        if not abv_cond.empty:
            results = results[results['Opponent'] == abv_cond.iloc[0]]

    results1 = results.copy()
    results2 = results.copy()

    results3 = results.copy()

    #r1: last 5, r2: last 10, r3: last season...before period filtering so that we can provide summariess
    results1 = results.sort_values('GAME_DATE', ascending = False).reset_index().iloc[:5, :]

    results2 = results.sort_values('GAME_DATE', ascending = False).reset_index().iloc[:9, :]

    results3 = resultsd[(resultsd['GAME_DATE'] >= pd.Timestamp('10/22/2024'))]

    results4 = results[(results['GAME_DATE'] >= pd.Timestamp('01/01/2020'))]

    all_avg = [results1, results2, results3, results4]
    

    

    # Store filtered results
    results_all = results.copy()

    # ✅ Define key stats to display
    all_results_avg = []
    title_avgs = ['Last 5 Games', 'Last 10 Games', 'This Season', 'Since 2020']
    for res, title in zip(all_avg, title_avgs):
        all_results_avg.append(get_avg_stats(res, title))

    percent_cols = ["FG_PCT", "FG3_PCT"]
    # for col in percent_cols:
    #     if col in avg_stats1.columns:
    #         avg_stats1[col] = avg_stats1[col].apply(lambda x: f"{float(x)*100:.1f}%" if x != 'nan' else "N/A")

    # ✅ Construct a dynamic title
    player_name = player if player else "All Players"
    opponent_name = opponent if opponent else "All Opponents"
    # date_range = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

    # title = f" 🏀 {player_name} vs {opponent_name}  \n📅 {date_range}"

    # ✅ Display Title in Markdown for Styling
    # display(Markdown(f"<div style='font-size:20px; font-weight:bold; color:#003366=;'>{title}</div>"))


    # ✅ Prepare detailed game logs

    results_all = pd.DataFrame(results_all, columns=["GAME_DATE", "Opponent",  
                                        "PTS", "REB", "AST", "P+R+A", "STL", "BLK", "TOV", "PLUS_MINUS", 
                                        "MIN", "FG_PCT", "FG3_PCT", "FANTASY_PTS"])

    # results_all = results_all.loc[:, ["PLAYER_NAME", "TEAM_NAME", "GAME_DATE", "Opponent",  
    #                                     "PTS", "REB", "AST", "P+R+A", "STL", "BLK", "TOV", "PLUS_MINUS", 
    #                                     "MIN", "FG_PCT", "FG3_PCT", "FANTASY_PTS"]].reset_index(drop=True)
    results_all['GAME_DATE'] = results_all['GAME_DATE'].dt.date
    # results_all['GAME_DATE'] = results_all['GAME_DATE'].strftime('%m/%d/%y')



    results_all = results_all.sort_values("GAME_DATE", ascending=False).reset_index().drop(columns = 'index')

    # ✅ Convert `GAME_DATE` to remove timestamps

    # ✅ Format percentage columns in game logs
    for col in percent_cols:
        if col in results_all.columns:
            results_all[col] = results_all[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")


    # ✅ Display Everything'
    # for i in all_results_avg:
    #     display(i)
    # display(styled_results)
    
    return [all_results_avg, results_all]


def on_click():
    return func_filter_nba(start_date, end_date, periods, player, opponent)


fetched_data = None

all_seasons_data = pull_nba_api_all_seasons("01/01/2020", datetime.now())



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    teams = sorted(get_nba_teams(), key=lambda x: x[0]) 
    players = pull_unique_vals()

    selected_t = None
    selected_p = None
    averages = None
    error_message = None  # Initialize error message

    if request.method == 'POST':
        selected_t = request.form.get('team')
        selected_p = request.form.get('player')

        # ✅ Check if player input is empty
        if not selected_p:
            error_message = "⚠️ Please enter a player name before submitting."
            return render_template('index.html', teams=teams, players=players, error_message=error_message)

        # ✅ Check if the player exists in the `players` list
        player_names = [player[0] for player in players]  # Extract player names
        if selected_p not in player_names:
            error_message = f"⚠️ '{selected_p}' is not a valid player. Please select from the list."
            return render_template('index.html', teams=teams, players=players, error_message=error_message)

        print(f"Selected Team: {selected_t}, Selected Player: {selected_p}")

        fetch_stats = func_filter_nba("01/01/2020", datetime.now(), None, selected_p, selected_t, all_seasons_data, players, teams)
        averages = fetch_stats[0]
        all_stats = fetch_stats[1].to_dict(orient="records")

        avg_dict0 = averages[0].to_dict(orient="records")[0]
        avg_dict1 = averages[1].to_dict(orient="records")[0]
        avg_dict2 = averages[2].to_dict(orient="records")[0]
        avg_dict3 = averages[3].to_dict(orient="records")[0]

        return render_template('index.html', teams=teams, players=players, avg_values0=avg_dict0, avg_values1=avg_dict1,
            avg_values2=avg_dict2, avg_values3=avg_dict3, selected_player_=selected_p, opp=selected_t, all_results=all_stats)

    return render_template('index.html', teams=teams, players=players, error_message=error_message, selected_player_=None)


if __name__ == '__main__':
    app.run(debug=True)


  #     except requests.exceptions.HTTPError as e:
    #         print(f"HTTP Error (Attempt {attempt + 1}/3): {e}")
    #         if attempt < 2:
    #             time.sleep(5)  # Wait before retrying
    #         else:
    #             return default_players  # Return default players after all attempts fail
    #     except requests.exceptions.RequestException as e:
    #         print(f"API request failed: {e}")
    #         return default_players  # Return default players on request failure
    #     except requests.exceptions.JSONDecodeError:
    #         print("Failed to parse JSON. The API may have blocked the request.")
    #         print("Response text:", response.text)
    #         return default_players  # Return default players on JSON parsing error

    # return default_players  # Fallback return in case of unexpected failure
