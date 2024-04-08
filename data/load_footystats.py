import requests
import pandas as pd
from utils.data_utils import Season
from data.clean import FootyStatsCleaner

API_KEY = "5949038c3c3fd7cf68cc60652121fa9e1aa460abc96871aee7a4ddabe097d87b"

class APIClient:
    """Football data API client"""

    def __init__(self, key):
        self.key = key
        self.BASE_URL = 'https://api.football-data-api.com'

    def _make_request(self, endpoint, params=None):
        """General method for making requests"""
        params = params or {}
        params = {'key': self.key, **params} 
        response = requests.get(f'{self.BASE_URL}/{endpoint}', params=params)
        response.raise_for_status()

        return response.json()['data']

class LeagueDataClient(APIClient):
    """API client for getting League data"""

    def get_league_list(self, **kwargs):
        return self._make_request('league-list', params=kwargs)

    def get_country_list(self):
        return self._make_request('country-list')

    def get_todays_matches(self, **kwargs):
        return self._make_request('todays-matches', params=kwargs)

    def get_league_matches(self, season_id, **kwargs):
        kwargs.update({'season_id': season_id})
        return self._make_request('league-matches', params=kwargs)

    def get_league_season(self, season_id, **kwargs):
        kwargs.update({'season_id': season_id})
        return self._make_request('league-season', params=kwargs)

    def get_league_teams(self, season_id, **kwargs):
        kwargs.update({'season_id': season_id})
        return self._make_request('league-teams', params=kwargs)

    def get_league_players(self, season_id, **kwargs):
        kwargs.update({'season_id': season_id})
        return self._make_request('league-players', params=kwargs)

    def get_league_referees(self, season_id, **kwargs):
        kwargs.update({'season_id': season_id})
        return self._make_request('league-referees', params=kwargs)

    def get_team(self, team_id):
        return self._make_request('team', {'team_id': team_id})
    
    def get_lastx(self, team_id):
        return self._make_request('lastx', {'team_id': team_id})

    def get_match_stats(self, match_id):
        return self._make_request('match', {'match_id': match_id})
    

class GenerateDataFrame():
    """General class for creating dataframes from API data"""
    
    def __init__(self, league_name="Premier League", country="England", season: Season=Season.Past1):
        self.client = LeagueDataClient(API_KEY)
        self.league_name = league_name
        self.country = country
        self.years = self.get_years(season)

    def get_years(self, season: Season):
        first_season = season.value.year
        last_season = 2023

        return [str(year) for year in range(first_season, last_season + 1)]


    def get_league_list_df(self):
        response = self.client.get_league_list()
        leagues_df = pd.DataFrame(response)
        return leagues_df

    def get_matches_by_league_df(self, season_id, max_per_page=None, page=None, max_time=None):
        params = {
            'max_per_page': max_per_page,
            'page': page,
            'max_time':max_time
        }
        response = self.client.get_league_matches(season_id, **params)
        matches_by_league_df = pd.DataFrame(response)
        return matches_by_league_df

    def get_filtered_leagues(self, league_name, country, years):
        leagues = self.get_league_list_df()
        filtered_leagues = leagues[(leagues['country'] == country) & (leagues['league_name'] == league_name)]
        
        all_leagues_data = []

        for _, row in filtered_leagues.iterrows():
            for season in row['season']:
                match_data = {
                    'id': season['id'],
                    'year': str(season['year'])[:4],
                    'league_name': row['league_name'],
                    'country': row['country']
                }
                all_leagues_data.append(match_data)
        
        all_leagues_data = pd.DataFrame(all_leagues_data)
        all_leagues_data = all_leagues_data[all_leagues_data['year'].isin(years)]

        return all_leagues_data

    def get_footystats_matches(self, league_name, country, years):
        leagues = self.get_filtered_leagues(league_name, country, years)
        all_matches = pd.DataFrame()

        for _, row in leagues.iterrows():
            matches = self.get_matches_by_league_df(row['id'])
            matches['season_id'] = row['id']
            matches['league_name'] = row['league_name']
            matches['country'] = row['country']
            matches['year'] = row['year']
            all_matches = pd.concat([all_matches, matches], ignore_index=True)

        return all_matches

    def get_referee_map_for_seasons(self, season_ids):
        referee_map = {}
        for season_id in season_ids:
            referees = self.client.get_league_referees(season_id)

            for ref in referees:
                referee_map[ref['id']] = ref['full_name']

        return referee_map

    def get_team_images_for_ids(self, team_ids):
        team_image_map = {}
        
        for team_id in team_ids:
            team_data = self.client.get_team(team_id)
            if team_data and 'image' in team_data[0]:
                team_image_map[team_id] = team_data[0]['image']
        
        return team_image_map
    
    def get_complete_matches(self, df: pd.DataFrame):
        # date less than 17/12/2023
        df = df[df['date'] <= '2023-12-17']
        return df[df['status'] == 'complete']
    
    def convert_and_sort_by_date(self, df: pd.DataFrame):
        df['date'] = pd.to_datetime(df['date_unix'], unit='s')
        df = df.sort_values('date')

        return df

    def load(self, is_save=False) -> pd.DataFrame:
        df = self.get_footystats_matches(self.league_name, self.country, self.years)

        if is_save:
            df.to_csv('raw_footystats.csv', index=False)
        
        df = self.convert_and_sort_by_date(df)
        df = self.get_complete_matches(df)
        df = FootyStatsCleaner(df).run()

        return df