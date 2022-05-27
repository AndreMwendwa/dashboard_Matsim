import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, mpld3
import pickle as pkl
import os
import sys
import seaborn as sns
import matplotlib.colors as colors
import sys
from textwrap import wrap
from pandas.plotting import parallel_coordinates
import datapane as dp
# from pyunpack import Archive
# from py7zr import unpack_7zarchive
import gzip
import shutil
import json
from collections import namedtuple
from dotmap import DotMap
# import shapefile as shp
import shapely.geometry
from shapely.geometry import Point
import PySimpleGUI as sg
from pathlib import Path
import yaml
import geopandas as gpd
import pickle as pkl
from collections import namedtuple
from matplotlib import style
style.use('seaborn')
import altair as alt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def select_file():
    '''Le GUI, pour séléctionner les dossiers avec les résultats'''
    form_rows = [[sg.Text('Please select the folder needed for the creation of the dashboard')],
                 [sg.Text('Scenario folder', size=(20, 1)),
                  sg.InputText(key='-file1-'), sg.FolderBrowse()],
                 [sg.Text('Shapefile', size=(20, 1)), sg.InputText(key='-file2-'),
                  sg.FileBrowse(target='-file2-')],
                 [sg.Text('Configuration file', size=(20, 1)), sg.InputText(key='-file3-'),
                  sg.FileBrowse(target='-file3-')],
                 [sg.Submit(), sg.Cancel()]]

    window = sg.Window('Choix de scénarios', form_rows)
    event, values = window.read()
    window.close()
    return event, values


def unzip_if_not_exists2(nom_simul):
    '''
        Pour unzipper le fichier des résultats si ça n'as pas encore été fait par le passé.
        '''
    isExist = os.path.exists(nom_simul)
    if not isExist:
        with gzip.open(nom_simul, 'rb') as f_in:
            with open(nom_simul[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.close()
            f_in.close()


class all_outputs:
    def __init__(self):
        # button, values = select_file()
        # f1, f2, f3 = values['-file1-'], values['-file2-'], values['-file3-']
        f1 = r'C:/Users/mwendwa.kiko/Documents/MATSim/send_MK/output_IdF_egt_10pct'
        f2 = 'C:/Users/mwendwa.kiko/Documents/MATSim/SaintDenis/SaintDenis.shp'
        f3 = 'C:/Users/mwendwa.kiko/Documents/MATSim/config_yml.yml'
        self.path_data = f1
        # unzip_if_not_exists2(Path(self.path_data + '/simulation_output_40iters/output_trips.csv.gz'))
        self.name_zone = f2.split('/')[-1].split('.')[0]  # Name of the zone of interest
        self.name_region = 'IdF'            # Name of the region of study
        # cache_path = '/'.join(f1.split('/')[:-1])         # Storing of the temporary file containing output_trips, if this
        # simulation has never been run before
        if not os.path.exists(f1 + f'/output_trips{self.name_zone}'):
            # If there is no file of results stored in the results folder with the name of the zone, it will create one
            self.output_trips = pd.read_csv(Path(self.path_data + '/simulation_output_40iters/output_trips.csv.gz'), sep=';')
            self.shapefile_folder = f2
            self.check_if_in_shape(14, 15, 'start')
            self.check_if_in_shape(18, 19, 'end')
            dbfile = open(f1 + f'/output_trips{self.name_zone}', 'wb')
            pkl.dump(self.output_trips, dbfile)
            dbfile.close()
        else:
            # else it will read what has been saved already
            dbfile = open(f1 + f'/output_trips{self.name_zone}', 'rb')
            self.output_trips = pkl.load(dbfile)
        # self.output_trips = (pd.read_csv(self.path_data + '/simulation_output_40iters/output_trips.csv', sep=';')
        #                      .sample(50000)
        #                      .reset_index()
        #                      .drop('index', axis=1)
        #                      )
        #TODO: Restore to correct state
        with open(f'{self.path_data}\\meta.json', 'r') as json_data:
            self.meta = json.load(json_data)
        self.output_trips.drop(self.output_trips[self.output_trips.traveled_distance == 0].index, inplace=True)

        yaml_file = open(f3, 'r')
        self.yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.modes = pd.DataFrame({'modes': ["walk", "bike", "pt", "car_passenger", "car"], 'mode_id':range(1, 6)})
        self.regional_local_mobility = DotMap()
        self.mobility_indicators_other_wd = self.mobility_indicators_other('whole_day')
        self.trips_RegLoc_Z_AAA_df = self.trips_RegLoc_Z_AAA()
        self.df_aggr_RegAgentLoc = self.f_aggregate_RegionAgentLocation()
        self.df_avg_tripAttrMotiveAgentLocation = self.f_avg_tripAttrMotiveAgentLocation()
        self.df_presenceAgent = self.f_presenceAgent()

    def scenario_config(self):
        scenario_config = pd.DataFrame(self.meta, index=[0])
        scenario_config.drop(['version', 'commit'], axis=1, inplace=True)
        # scenario_config.index = scenario_config['sampling_rate']
        # scenario_config.drop('sampling_rate', axis=1, inplace=True)
        return scenario_config

    def graphs_mobility(self, df_input, name):
        chart = alt.Chart(df_input.T.iloc[1:].reset_index()).mark_bar(size=100, y=-30).encode(
            x=alt.X(f'sum({name})', axis=alt.Axis(grid=False)),
            # y='index',
            color=alt.Color('index:N', scale=alt.Scale(
                domain=['work_trips', 'home_trips', 'education_trips', 'shop_trips', 'leisure_trips', 'other_trips'],
                range=['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373', 'violet']))
        )
        return chart

    # def graphs_mobility2(self, df_input, name):
    #     rolling_sum = 0     # The values that we will plot in the bar graphs
    #     plot_values = {}
    #     for col in df_input.columns.to_list()[1:]:
    #         rolling_sum += df_input.loc[name, col]
    #         plot_values[col] = rolling_sum
    #     fig, ax = plt.subplots(figsize=(5, 3))
    #     colors_used = ['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373', 'violet']
    #     i = 0   # Counter for the colors
    #     for key, value in reversed(plot_values.items()):
    #         sns.barplot(x=[' '], y=[value], color=colors_used[i], label=key)
    #         i += 1
    #     ax.legend(loc="right", frameon=True)
    #     plt.title(f'Bar Plot of {name} trips')
    #     return fig

    def mobility_indicators_total(self, period):
        if period == 'whole_day':
            trips = self.output_trips.copy()
        else:
            trips = self.select_period(period)
        total_population = self.output_trips['person'].unique().size
        total_population_df = {}  # On créet un dictionnaire d'abord pour ensuite transformer en dataframe
        total_population_df['Total_trips'] = trips.shape[0]
        total_population_df['work_trips'] = trips[(trips['end_activity_type'] == 'work')].shape[0]
        total_population_df['home_trips'] = trips[(trips['end_activity_type'] == 'home')].shape[0]
        total_population_df['education_trips'] = trips[(trips['end_activity_type'] == 'education')].shape[0]
        total_population_df['shop_trips'] = trips[(trips['end_activity_type'] == 'shop')].shape[0]
        total_population_df['leisure_trips'] = trips[(trips['end_activity_type'] == 'leisure')].shape[0]
        total_population_df['other_trips'] = trips[(trips['end_activity_type'] == 'other')].shape[0]
        total_population_df = pd.DataFrame(total_population_df, index=['whole_population'])
        total_graph = self.graphs_mobility(total_population_df, 'whole_population')
        return [total_population, total_population_df, total_graph]

    def mobility_indicators_other(self, period):
        def dataframe_gen(mask, name):
            regional_df = {}  # On créet un dictionnaire d'abord pour ensuite transformer en dataframe
            regional_df['Total_trips'] = (mask).sum()
            regional_df['work_trips'] = (trips.loc[(mask), 'end_activity_type'] == 'work').sum()
            regional_df['home_trips'] = (trips.loc[(mask), 'end_activity_type'] == 'home').sum()
            regional_df['education_trips'] = (
                    trips.loc[(mask), 'end_activity_type'] == 'education').sum()
            regional_df['shop_trips'] = (trips.loc[mask, 'end_activity_type'] == 'shop').sum()
            regional_df['leisure_trips'] = (
                    trips.loc[mask, 'end_activity_type'] == 'leisure').sum()
            regional_df['other_trips'] = (
                    trips.loc[mask, 'end_activity_type'] == 'other').sum()
            regional_df = pd.DataFrame(regional_df, index=[f'{name}_regional'])
            # mobility_indicators[f'{name}_df'] = regional_df
            mobility_indicators[f'{name}_regional'] = regional_df.copy()

            local_df = {}  # On créet un dictionnaire d'abord pour ensuite transformer en dataframe
            local_df['Total_trips'] = (mask & mask_fully_in_zone).sum()
            local_df['work_trips'] = (trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'work').sum()
            local_df['home_trips'] = (trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'home').sum()
            local_df['education_trips'] = (
                    trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'education').sum()
            local_df['shop_trips'] = (trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'shop').sum()
            local_df['leisure_trips'] = (
                    trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'leisure').sum()
            local_df['other_trips'] = (
                    trips.loc[(mask & mask_fully_in_zone), 'end_activity_type'] == 'other').sum()
            local_df = pd.DataFrame(local_df, index=[f'{name}_local'])
            mobility_indicators[f'{name}_local'] = local_df.copy()        
        
        mobility_indicators = DotMap()  # Dictionary of return values

        if period == 'whole_day':
            trips = self.output_trips.copy()
        else:
            trips = self.select_period(period)
        # trips = trips_in.copy()
        trips_reg = trips.loc[(trips['In_Zone_start'] == 1) | (trips['In_Zone_end'] == 1)]

        mask_in_zone = (trips['In_Zone_start'] == 1) | (trips['In_Zone_end'] == 1)  # Trips that happen strictly in the
        # zone; useful for defining commuter and tourist trips.
        mask_fully_in_zone = (trips['In_Zone_start'] == 1) & (
                    trips['In_Zone_end'] == 1)  # Trips that happen strictly in the
        # zone; useful for defining local trips.

        # Residents
        mask_res_in_zone = (
                ((trips['In_Zone_start'] == 1) & (trips['start_activity_type'] == 'home')) |
                ((trips['In_Zone_end'] == 1) & (trips['end_activity_type'] == 'home'))
        )  # Les lignes seulement qui appartiennent à la zone en question
        residents_population = set(trips.loc[mask_res_in_zone, 'person'])

        mask_res = trips['person'].isin(residents_population)
        residents_population = set(trips.loc[mask_res_in_zone, 'person'])
        mobility_indicators.residents_population = len(residents_population)
        
        dataframe_gen(mask_res, 'residents')
        self.regional_local_mobility.residents_regional = trips.loc[(mask_res)]       # Used later in mode share calc
        self.regional_local_mobility.residents_local = trips.loc[(mask_res & mask_fully_in_zone)]    # Used in mode share calc
        mobility_indicators.residents_regional_graph = self.graphs_mobility(mobility_indicators.residents_regional,
                                                                   'residents_regional')
        # We draw the bar graph that we will later add to the report
        mobility_indicators.residents_local_graph = self.graphs_mobility(mobility_indicators.residents_local, 'residents_local')
        # We draw the bar graph that we will later add to the report

        # Commuters
        mask_comm_in_zone = (
                ((trips['In_Zone_start'] == 1) & (trips['start_activity_type'] == 'work')) |
                ((trips['In_Zone_end'] == 1) & (trips['end_activity_type'] == 'work'))
        )  # Les lignes seulement qui appartiennent à la zone en question
        commuters_population = set(trips.loc[mask_comm_in_zone, 'person']).difference(residents_population)
        mobility_indicators.commuters_population = len(commuters_population)

        mask_comm = trips['person'].isin(commuters_population)
        dataframe_gen((mask_comm & mask_in_zone), 'commuters')
        self.regional_local_mobility.commuters_regional = trips.loc[(mask_comm & mask_in_zone)]  # Used later in mode share calc
        self.regional_local_mobility.commuters_local = trips.loc[((mask_comm & mask_in_zone) & mask_fully_in_zone)]  # Used in mode share calc
        mobility_indicators.commuters_regional_graph = self.graphs_mobility(mobility_indicators.commuters_regional,
                                                                   'commuters_regional')
        # We draw the bar graph that we will later add to the report
        mobility_indicators.commuters_local_graph = self.graphs_mobility(mobility_indicators.commuters_local,
                                                                   'commuters_local')
        # We draw the bar graph that we will later add to the report

        # Tourists
        tourists_population = (
            set(trips.loc[mask_in_zone, 'person'])
                .difference(residents_population.union(commuters_population))
        )
        mobility_indicators.tourists_population = len(tourists_population)
        mask_tourists = trips['person'].isin(tourists_population)
        dataframe_gen((mask_tourists & mask_in_zone), 'tourists')
        self.regional_local_mobility.tourists_regional = trips.loc[(mask_tourists & mask_in_zone)]  # Used later in mode share calc
        self.regional_local_mobility.tourists_local = trips.loc[((mask_tourists & mask_in_zone) & mask_fully_in_zone)]  # Used in mode share calc
        mobility_indicators.tourists_regional_graph = self.graphs_mobility(mobility_indicators.tourists_regional,
                                                                   'tourists_regional')
        # We draw the bar graph that we will later add to the report
        mobility_indicators.tourists_local_graph = self.graphs_mobility(mobility_indicators.tourists_local,
                                                                   'tourists_local')

        # We draw the bar graph that we will later add to the report
        mobility_indicators.residents_df = pd.concat([mobility_indicators.residents_regional, 
                                                      mobility_indicators.residents_local])
        mobility_indicators.commuters_df = pd.concat([mobility_indicators.commuters_regional,
                                                      mobility_indicators.commuters_local])
        mobility_indicators.tourists_df = pd.concat([mobility_indicators.tourists_regional,
                                                      mobility_indicators.tourists_local])
        return mobility_indicators

    def select_period(self, period):
        trips = self.output_trips.copy()
        tmp1 = trips['dep_time']
        tmp2 = tmp1.str.split(':', expand=True)
        tmp2[0] = tmp2[0].astype('int64')
        tmp2[0] = np.where(tmp2[0] >= 24, tmp2[0] - 24, tmp2[0])
        tmp2[0] = tmp2[0].astype('str')
        tmp3 = tmp2[0].str.cat([tmp2[1], tmp2[2]], sep=':')
        departure = pd.to_timedelta(tmp3)
        trav = pd.to_timedelta(trips.loc[:, 'trav_time'])
        arrival = trav + departure

        period = self.yaml_content[period]
        mask = (departure > pd.to_timedelta(f'{period[0]}:00:00')) & (
                    departure < pd.to_timedelta(f'{period[1]}:00:00'))  # To be replaced with variables for times we want to look at.
        trips_masked = trips.loc[mask, :]
        return trips_masked

    def trips_RegLoc_Z_AAA(self):
        mobility_indicators = self.regional_local_mobility
        mobility_indicators.residents_regional['typeAgent'] = 'Residents'
        mobility_indicators.residents_regional['typeLocation'] = 'Regional'
        mobility_indicators.commuters_regional['typeAgent'] = 'Commuters'
        mobility_indicators.commuters_regional['typeLocation'] = 'Regional'
        mobility_indicators.tourists_regional['typeAgent'] = 'Tourists'
        mobility_indicators.tourists_regional['typeLocation'] = 'Regional'

        mobility_indicators.residents_local['typeAgent'] = 'Residents'
        mobility_indicators.residents_local['typeLocation'] = 'Local'
        mobility_indicators.commuters_local['typeAgent'] = 'Commuters'
        mobility_indicators.commuters_local['typeLocation'] = 'Local'
        mobility_indicators.tourists_local['typeAgent'] = 'Tourists'
        mobility_indicators.tourists_local['typeLocation'] = 'Local'
        trips_RegLoc_Z_AAA = pd.concat([mobility_indicators.residents_regional, mobility_indicators.commuters_regional,
                                        mobility_indicators.tourists_regional, mobility_indicators.residents_local,
                                        mobility_indicators.commuters_local, mobility_indicators.tourists_local])
        trips_RegLoc_Z_AAA['Region'] = self.name_zone
        return trips_RegLoc_Z_AAA.reset_index(drop=True)

    def f_count_tripAttribut(self, df, region, typeAgent, typeLocation):
        df_trips = (df[['longest_distance_mode', 'traveled_distance']].groupby(by='longest_distance_mode')
                    .count().rename(columns={'traveled_distance': 'mode_share'}))   # Trips
        df_trips['tripAttr'] = "Trips"
        df_trips['pct'] = df_trips['mode_share']/df_trips['mode_share'].sum()         # Adding a % colm

        df_dist = (df[['longest_distance_mode', 'traveled_distance']].groupby(by='longest_distance_mode')
            .sum('traveled_distance').rename(columns={'traveled_distance': 'mode_share'}))           # Distance
        df_dist['pct'] = df_dist['mode_share'] / df_dist['mode_share'].sum()          # Adding a % colm
        df_dist['tripAttr'] = "Distance"

        df['trav_time'] = pd.to_timedelta(df['trav_time']).dt.seconds / 60      # Convert travel times to seconds
        df_time = (df[['longest_distance_mode', 'trav_time']].groupby(by='longest_distance_mode')
                   .sum().rename(columns={'trav_time': 'mode_share'}))      # Travel Time
        df_time['pct'] = df_time['mode_share'] / df_time['mode_share'].sum()      # Adding a % colm
        df_time['tripAttr'] = "TTime"
        # Put together and add percentage as a string
        df_aggr = pd.concat([df_trips, df_dist, df_time])
        df_aggr = pd.merge(df_aggr, self.modes, left_index=True, right_on='modes')
        df_aggr.sort_values(by='mode_id', inplace=True)         # Sort by descending Mode_id
        df_aggr[['Region', 'typeAgent', 'typeLocation']] = region, typeAgent, typeLocation       # Add Region
        return df_aggr

    # Modal share by Trips/Distance for region,typeAgent,typeLocation
    def f_aggregate_RegionAgentLocation(self):
        df_aggr_RegAgentLoc = pd.concat([
            # RZ / Z - All agents, All trips
            self.f_count_tripAttribut(self.output_trips, region=self.name_region, typeAgent='All', typeLocation='All'),
            self.f_count_tripAttribut(self.trips_RegLoc_Z_AAA_df, region=self.name_zone, typeAgent='All', typeLocation='All'),
            # Regional trips
            self.f_count_tripAttribut(self.regional_local_mobility.residents_regional, region=self.name_zone,
                                      typeAgent='Residents', typeLocation='Regional'),
            self.f_count_tripAttribut(self.regional_local_mobility.commuters_regional, region=self.name_zone,
                                      typeAgent='Commuters', typeLocation='Regional'),
            self.f_count_tripAttribut(self.regional_local_mobility.tourists_regional, region=self.name_zone,
                                      typeAgent='Tourists', typeLocation='Regional'),
            # Local trips
            self.f_count_tripAttribut(self.regional_local_mobility.residents_local, region=self.name_zone,
                                      typeAgent='Residents', typeLocation='Local'),
            self.f_count_tripAttribut(self.regional_local_mobility.commuters_local, region=self.name_zone,
                                      typeAgent='Commuters', typeLocation='Local'),
            self.f_count_tripAttribut(self.regional_local_mobility.tourists_local, region=self.name_zone,
                                      typeAgent='Tourists', typeLocation='Local')
        ])
        df_aggr_RegAgentLoc.index = df_aggr_RegAgentLoc['modes']
        return df_aggr_RegAgentLoc

    def f_avg_tripAttrMotiveAgentLocation(self):
        df = self.trips_RegLoc_Z_AAA_df.copy()
        df = df.loc[df['end_activity_type'] != 'other']
        # Distance
        df_dist = pd.DataFrame(
            df.groupby(by=['typeAgent', 'typeLocation', 'end_activity_type']).mean()['traveled_distance']
        ).rename(columns={'traveled_distance': 'avg'})/ 1000
        df_dist['tripAttr'] = 'Distance'        # New column for name of attribute
        # df['trav_time'] = pd.to_timedelta(df.loc[:, 'trav_time']).dt.seconds / 60       # Travel Time
        df_ttime = pd.DataFrame(
            df.groupby(by=['typeAgent', 'typeLocation', 'end_activity_type']).mean()['trav_time']
        ).rename(columns={'trav_time': 'avg'})
        df_ttime['tripAttr'] = 'TTime'          # New column for name of attribute
        df_avg = pd.concat([df_dist, df_ttime])
        df_avg['AgentLocation'] = [f'{item[0]}_{item[1]}' for item in list(df_avg.index)]
        return df_avg.reset_index()


    def f_presenceAgent(self):
        df = self.trips_RegLoc_Z_AAA_df.copy()
        df['dep_time'] = pd.to_timedelta(df['dep_time']).dt.round('5min')
        df['arr_time'] = (pd.to_timedelta(df['dep_time']) + pd.to_timedelta(df['wait_time']) +
                          pd.to_timedelta(df['trav_time'])).dt.round('5min')
        
        # Number coming out
        df_out = df.loc[(df['In_Zone_start'] == 1) & (df['In_Zone_end'] != 1)]
        df_out = pd.DataFrame(df_out.groupby(by=['dep_time', 'typeAgent']).size()).rename(columns={0 : 'nb_out'})
        df_out['Time'] = [item[0] for item in df_out.index]
        df_out['typeAgent_col'] = [item[1] for item in df_out.index]

        # Number coming in
        df_in = df.loc[(df['In_Zone_start'] != 1) & (df['In_Zone_end'] == 1)]
        df_in = pd.DataFrame(df_in.groupby(by=['dep_time', 'typeAgent']).size()).rename(columns={0 : 'nb_in'})
        df_in['Time'] = [item[0] for item in df_in.index]
        df_in['typeAgent_col'] = [item[1] for item in df_in.index]

        df_in_out = pd.merge(df_in, df_out, on=['Time', 'typeAgent_col'], how='outer')      # Merge df's, on time and
        # typeAgent
        df_in_out[['nb_out', 'nb_in']] = df_in_out[['nb_out', 'nb_in']].fillna(0).copy()
        # df_in_out = (
        #     df_in_out.sort_values(by=['Time', 'typeAgent_col']).reset_index(drop=True).groupby(by='typeAgent_col')
        #     .cumsum()
        # )
        # df_in_out = (
        #     df_in_out.sort_values(by=['Time', 'typeAgent_col']).reset_index(drop=True).groupby(by='typeAgent_col')
        # )

        # Cumsum for residents, commuters, tourists separately, because these labels are lost if it's done under the
        # groupby
        # Residents
        df_res = df_in_out.loc[df_in_out['typeAgent_col'] == 'Residents'].copy().sort_values(by='Time')
        df_res.loc[:, ('nb_in', 'nb_out')] = df_res.loc[:, ('nb_in', 'nb_out')].cumsum()
        df_res['nb_cum'] = df_res['nb_in'] - df_res['nb_out'] + self.mobility_indicators_other_wd.residents_population
        # Commuters
        df_comm = df_in_out.loc[df_in_out['typeAgent_col'] == 'Commuters'].copy().sort_values(by='Time')
        df_comm.loc[:, ('nb_in', 'nb_out')] = df_comm.loc[:, ('nb_in', 'nb_out')].cumsum()
        df_comm['nb_cum'] = df_comm['nb_in'] - df_comm['nb_out']

        # Tourists
        df_tour = df_in_out.loc[df_in_out['typeAgent_col'] == 'Tourists'].copy().sort_values(by='Time')
        df_tour.loc[:, ('nb_in', 'nb_out')] = df_tour.loc[:, ('nb_in', 'nb_out')].cumsum()
        df_tour['nb_cum'] = df_tour['nb_in'] - df_tour['nb_out']

        df_combine = pd.concat([df_res, df_tour, df_comm]).reset_index(drop=True).sort_values(by='Time')

        # df_result = pd.merge(df_res[['Time', 'nb_cum']], df_comm[['Time', 'nb_cum']],
        #     on='Time', how='outer').rename(columns={'nb_cum_x': 'nb_cum_res', 'nb_cum_y': 'nb_cum_comm'})
        # df_result = pd.merge(df_result, df_tour[['Time', 'nb_cum']],
        #     on='Time', how='outer').rename(columns={'nb_cum': 'nb_cum_tour'}).sort_values(by='Time')
        # df_result.loc[:, ('nb_cum_res', 'nb_cum_comm', 'nb_cum_tour')] = \
        #     df_result.loc[:, ('nb_cum_res', 'nb_cum_comm', 'nb_cum_tour')].fillna(0)
        # df_result['nb_cum_res'] += self.mobility_indicators_other_wd.residents_population
        # df_result['nb_cum'] = df_result['nb_cum_res'] + df_result['nb_cum_comm'] + df_result['nb_cum_tour']
        return df_combine

    def plot_presence(self):
        df_presence = self.df_presenceAgent.copy()
        df_presence['Hour'] = df_presence['Time'].dt.days * 24 + df_presence['Time'].dt.seconds / 3600
        df_presence.drop('Time', axis=1, inplace=True)

        # An additional column is needed for sorting
        df_presence['sort_id'] = np.where(df_presence['typeAgent_col'] == 'Residents', 1,
                                          np.where(df_presence['typeAgent_col'] == 'Commuters', 2, 3))
        chart = alt.Chart(df_presence).mark_area().encode(
            x='Hour:Q', y=alt.Y('nb_cum:Q', stack='zero'), color='typeAgent_col', order=alt.Order('sort_id', sort='ascending')
        )
        return chart

    # def plot_presence(self):
    #     df_presence = self.df_presenceAgent.copy()
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot()
    #     sns.lineplot(ax=ax1, x=df_presence['Time'].dt.round('30min'), y=df_presence['nb_cum_res'])
    #     sns.lineplot(ax=ax1, x=df_presence['Time'].dt.round('30min'), y=df_presence['nb_cum_res'] +
    #                                                                     df_presence['nb_cum_comm'])
    #     sns.lineplot(ax=ax1, x=df_presence['Time'].dt.round('30min'),
    #                  y=df_presence['nb_cum_res'] + df_presence['nb_cum_comm'] + df_presence['nb_cum_tour'])
    #     ax1.set_ylabel('Number of people')
    #     return fig

    # def plot_distance_motive(self):
    #     df = self.df_avg_tripAttrMotiveAgentLocation.copy()
    #     df.index = df['typeAgent']
    #     fig, axes = plt.subplots(5, 2)
    #     i = 0       # counter for axes
    #     for typeLoc in ['Local', 'Regional']:
    #         for end_act in ['leisure', 'shop', 'work', 'education', 'home']:
    #             mask = (df['tripAttr'] == 'Distance') & (df['typeLocation'] == typeLoc) & (df['end_activity_type'] == end_act)
    #             sns.barplot(y=df.loc[mask, ('typeAgent')], x=df.loc[mask, ('avg')], ax=axes[i])
    #             i += 1
    #     return fig

    def plot_distance_motive(self):
        df = self.df_avg_tripAttrMotiveAgentLocation.copy()
        chart1a = alt.Chart(df.loc[(df['typeLocation'] == 'Local') & (df['tripAttr'] == 'Distance')], width=500,
                            height=alt.Step(8), title='Local Trips').mark_bar().encode(
            y=alt.Y("typeAgent:N", axis=None),
            x=alt.X("avg:Q", title='Distance (km)', axis=alt.Axis(), sort='descending', scale={'domain': [0, 20]}),
            color=alt.Color(
                "typeAgent:N", title="Agent Types", legend=alt.Legend(orient="bottom", titleOrient="left")
            ),
            row=alt.Row("end_activity_type:N", title="Trip Motives", header=alt.Header(labelAngle=0)),
            
        )

        chart1b = alt.Chart(df.loc[(df['typeLocation'] == 'Regional') & (df['tripAttr'] == 'Distance')], width=500,
                            height=alt.Step(8), title='Regional Trips').mark_bar().encode(
            y=alt.Y("typeAgent:N", axis=None),
            x=alt.X("avg:Q", title='Distance (km)', axis=alt.Axis(), scale={'domain': [0, 20]}),
            color=alt.Color(
                "typeAgent:N", title="Agent Types", legend=alt.Legend(orient="bottom", titleOrient="left")
            ),
            row=alt.Row("end_activity_type:N", title=None, header=alt.Header(labels=False)),
            
        )
        
        # chart1 = chart1a | chart1b
        chart1 = alt.hconcat(chart1a, chart1b)

        # chart2a = alt.Chart(df.loc[(df['typeLocation'] == 'Local') & (df['tripAttr'] == 'Time')], width=500,
        #                     height=alt.Step(8)).mark_bar().encode(
        #     y=alt.Y("typeAgent:N", axis=None),
        #     x=alt.X("avg:Q", title=None, axis=alt.Axis(), sort='descending', scale={'domain': [0, 55]}),
        #     color=alt.Color(
        #         "typeAgent:N", title="Agent Types", legend=alt.Legend(orient="bottom", titleOrient="left")
        #     ),
        #     row=alt.Row("end_activity_type:N", title="Trip Motives", header=alt.Header(labelAngle=0)),
        #
        # )
        #
        # chart2b = alt.Chart(df.loc[(df['typeLocation'] == 'Regional') & (df['tripAttr'] == 'Time')], width=500,
        #                     height=alt.Step(8)).mark_bar().encode(
        #     y=alt.Y("typeAgent:N", axis=None),
        #     x=alt.X("avg:Q", title=None, axis=alt.Axis(), scale={'domain': [0, 55]}),
        #     color=alt.Color(
        #         "typeAgent:N", title="Agent Types", legend=alt.Legend(orient="bottom", titleOrient="left")
        #     ),
        #     row=alt.Row("end_activity_type:N", title=None, header=alt.Header(labels=False)),
        #
        # )
        #
        # # chart2 = chart2a | chart2b
        # chart2 = alt.hconcat(chart2a, chart2b)

        # chart = chart1 & chart2
        return chart1

    # def barplot2(self, mask_in, colors_used, name, fig):
    #     mask = (
    #             (self.df_aggr_RegAgentLoc['typeAgent'] == mask_in.typeAgent) &
    #             (self.df_aggr_RegAgentLoc['Region'] == mask_in.Region) &
    #             (self.df_aggr_RegAgentLoc['tripAttr'] == mask_in.tripAttr) &
    #             (self.df_aggr_RegAgentLoc['typeLocation'] == mask_in.typeLocation)
    #     )
    #     trips_plot = self.df_aggr_RegAgentLoc.loc[mask].copy()
    #     rolling_sum = 0  # The values that we will plot in the bar graphs
    #     plot_values = {}
    #     # plot_text = [0]     # Counter for the text to be plotted.
    #     for mode in self.modes['modes']:
    #         rolling_sum += trips_plot.loc[mode, 'pct']
    #         plot_values[mode] = rolling_sum
    #         # plot_text.append(rolling_sum)
    #     ax = fig.add_subplot(figsize=(8, 3))
    #     # colors_used = ['#CDE2CD', '#90D2DA', '#F4E6BE', '#EFB4A2', '#D48484']
    #     i = 0  # Counter for the colors
    #     for key, value in reversed(plot_values.items()):
    #         sns.barplot(x=[' '], y=[value], color=colors_used[i], label=key)
    #         # plt.text(plot_text[i] + plot_text[i + 1])
    #         i += 1
    #     ax.legend(loc="right", frameon=True)
    #     plt.title(f'Bar Plot of {name} trips')
    #     return fig

    # def barplot(self, mask_in, colors_used, name):
    #     mask = (
    #             (self.df_aggr_RegAgentLoc['typeAgent'] == mask_in.typeAgent) &
    #             (self.df_aggr_RegAgentLoc['Region'] == mask_in.Region) &
    #             (self.df_aggr_RegAgentLoc['tripAttr'] == mask_in.tripAttr) &
    #             (self.df_aggr_RegAgentLoc['typeLocation'] == mask_in.typeLocation)
    #     )
    #     trips_plot = self.df_aggr_RegAgentLoc.loc[mask].copy()
    #     rolling_sum = 0  # The values that we will plot in the bar graphs
    #     plot_values = {}
    #     # plot_text = [0]     # Counter for the text to be plotted.
    #     for mode in self.modes['modes']:
    #         rolling_sum += trips_plot.loc[mode, 'pct']
    #         plot_values[mode] = rolling_sum
    #         # plot_text.append(rolling_sum)
    #     fig, ax = plt.subplots(figsize=(5, 3))
    #     # colors_used = ['#CDE2CD', '#90D2DA', '#F4E6BE', '#EFB4A2', '#D48484']
    #     i = 0  # Counter for the colors
    #     for key, value in reversed(plot_values.items()):
    #         sns.barplot(x=[' '], y=[value], color=colors_used[i], label=key)
    #         # plt.text(plot_text[i] + plot_text[i + 1])
    #         i += 1
    #     ax.legend(loc="right", frameon=True)
    #     plt.title(f'Bar Plot of {name} trips')
    #     return fig

    # def barplot3(self, mask_in, colors_used, name):
    #     mask = (
    #             (self.df_aggr_RegAgentLoc['typeAgent'] == mask_in.typeAgent) &
    #             (self.df_aggr_RegAgentLoc['Region'] == mask_in.Region) &
    #             (self.df_aggr_RegAgentLoc['tripAttr'] == mask_in.tripAttr) &
    #             (self.df_aggr_RegAgentLoc['typeLocation'] == mask_in.typeLocation)
    #     )
    #     trips_plot = self.df_aggr_RegAgentLoc.loc[mask].copy()
    #     rolling_sum = 0  # The values that we will plot in the bar graphs
    #     plot_values = {}
    #     for mode in self.modes['modes']:
    #         rolling_sum += trips_plot.loc[mode, 'pct']
    #         plot_values[mode] = rolling_sum
    #     fig, ax = plt.subplots(figsize=(5, 3))
    #     # colors_used = ['#CDE2CD', '#90D2DA', '#F4E6BE', '#EFB4A2', '#D48484']
    #     i = 0  # Counter for the colors
    #     for key, value in reversed(plot_values.items()):
    #         sns.barplot(x=[' '], y=[value], color=colors_used[i], label=key)
    #         i += 1
    #     ax.legend(loc="right", frameon=True)
    #     plt.title(f'Bar Plot of {name} trips')
    #     return fig

    def check_if_in_shape(self, col_x, col_y, name):
        trips = self.output_trips.copy()
        Z = gpd.read_file(self.shapefile_folder)         # Read shapefile of zone Z
        Z['dummy'] = 'dummy'                # add dummy column to dissolve all geometries into one
        geom_Z = Z.dissolve(by='dummy').geometry[0]  # take the single union geometry
        gdf_RZ = gpd.points_from_xy(trips.iloc[:, col_x], trips.iloc[:, col_y])
        self.output_trips.loc[gdf_RZ.within(geom_Z), f'In_Zone_{name}'] = 1
        self.output_trips[f'In_Zone_{name}'].fillna(0, inplace=True)


        # sf = shp.Reader(self.shapefile_folder)
        # shape = shapely.geometry.asShape(sf.shape(0))
        # minx, miny, maxx, maxy = shape.bounds
        # bounding_box = shapely.geometry.box(minx, miny, maxx, maxy)
        # # trips['In_zone'] = 0
        # In_Zone = pd.Series(np.zeros(trips.shape[0]))
        #
        # i = 0    # Counter needed here
        # for line in trips.itertuples():
        #     pt = shapely.geometry.Point(int(line[col_x + 1]), int(line[col_y + 1]))     # We add 1 because of the
        #     # itertuples
        #     if bounding_box.contains(pt):
        #         if shape.contains(pt):
        #             In_Zone[i] = 1
        #     i += 1
        # In_Zone.index = self.output_trips.index
        # self.output_trips[f'In_Zone_{name}'] = In_Zone


    # def check_if_in_shape2(self, col_x, col_y, name):
    #     trips = self.output_trips.copy()
    #     sf = shp.Reader(self.shapefile_folder)
    #     shape = shapely.geometry.asShape(sf.shape(0))
    #     minx, miny, maxx, maxy = shape.bounds
    #     bounding_box = shapely.geometry.box(minx, miny, maxx, maxy)
    #     # trips['In_zone'] = 0
    #     In_Zone = pd.Series(np.zeros(trips.shape[0]))
    #
    #     i = 0    # Counter needed here
    #     for line in trips.itertuples():
    #         pt = shapely.geometry.Point(int(line[col_x + 1]), int(line[col_y + 1]))     # We add 1 because of the
    #         # itertuples
    #         if bounding_box.contains(pt):
    #             if shape.contains(pt):
    #                 In_Zone[i] = 1
    #         i += 1
    #     In_Zone.index = self.output_trips.index
    #     self.output_trips[f'In_Zone_{name}'] = In_Zone
    #     # In_Zone.to_csv(f'In_Zone_{name}')


def dashboard_datapane_matsim(scenario):
    mobility_indicators_other_whole_day = scenario.mobility_indicators_other('whole_day')
    mobility_indicators_other_PPM = scenario.mobility_indicators_other('PPM')
    mobility_indicators_other_PCJ = scenario.mobility_indicators_other('PCJ')
    mobility_indicators_other_PPS = scenario.mobility_indicators_other('PPS')

    # Mode Share Bar Charts
    colLoc = ['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373']
    colReg = ['#CDE2CD', '#90D2DA', '#F4E6BE', '#EFB4A2', '#D48484']
    mask = namedtuple('mask', 'typeAgent Region tripAttr typeLocation')


    df_aggr_RegAgentLoc = scenario.df_aggr_RegAgentLoc.copy()
    chart1 = alt.Chart(df_aggr_RegAgentLoc.loc[df_aggr_RegAgentLoc['typeLocation'] == 'Regional'], width=20,
                       height=alt.Step(100)).mark_bar(size=100).encode(
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format="%", labelFontSize=13), scale={'domain': [0, 1]}, title='Cumulative Percentage'),
        x=alt.X('tripAttr:N', axis=alt.Axis(labelFontSize=20), title='Trip Attribute Considered'),
        color=alt.Color('modes:N', scale=alt.Scale(
        domain=['walk', 'bike', 'pt', 'car_passenger', 'car'],
            range=['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373']),
        legend=alt.Legend(orient="bottom", titleOrient="left", labelFontSize=20, title='Modes', titleFontSize=15)),
        row=alt.Row('typeAgent:N', header=alt.Header(labelFontSize=20), title='Type of Individual'),
        column=alt.Column('typeLocation:N', header=alt.Header(labelFontSize=20), title=None),

    )

    chart2 = alt.Chart(df_aggr_RegAgentLoc.loc[(df_aggr_RegAgentLoc['typeLocation'] == 'Local')], width=20,
                       height=alt.Step(100)).mark_bar(size=100).encode(
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format="%", labelFontSize=13), scale={'domain': [0, 1]}, title='Cumulative Percentage'),
        x=alt.X('tripAttr:N', axis=alt.Axis(labelFontSize=20), title='Trip Attribute Considered'),
        color=alt.Color('modes:N', scale=alt.Scale(
        domain=['walk', 'bike', 'pt', 'car_passenger', 'car'],
            range=['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373']),
        legend=alt.Legend(orient="bottom", titleOrient="left", labelFontSize=20, title='Modes', titleFontSize=15)),
        row=alt.Row('typeAgent:N', header=alt.Header(labelFontSize=20), title='Type of Individual'),
        column=alt.Column('typeLocation:N', header=alt.Header(labelFontSize=20), title=None),

    )

    # TODO: Divided by 2 in the line below, but that is only a temporary fix.
    chart3 = alt.Chart(df_aggr_RegAgentLoc.loc[(df_aggr_RegAgentLoc['typeLocation'] == 'All')], width=20,
                       height=alt.Step(100)).mark_bar(size=100).encode(
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format="%", labelFontSize=13), scale={'domain': [0, 1]}, title='Cumulative Percentage'),
        x=alt.X('tripAttr:N', axis=alt.Axis(labelFontSize=20), title='Trip Attribute Considered'),
        color=alt.Color('modes:N', scale=alt.Scale(
        domain=['walk', 'bike', 'pt', 'car_passenger', 'car'], range=['#C8E6C9', '#80DEEA', '#FFECB3', '#FFAB91', '#E57373']),
        legend=alt.Legend(orient="bottom", titleOrient="left", labelFontSize=20, title='Modes', titleFontSize=15))
        , row=alt.Row('Region:N', header=alt.Header(labelFontSize=20), title=None),
        column=alt.Column('typeLocation:N', header=alt.Header(labelFontSize=20), title=None),

    )

    # Plots of average distance and time for different motives by different kinds of agents
    # distplot, timeplot = scenario.plot_distance_motive()
    distplot = scenario.plot_distance_motive()

    report = dp.Report(
        dp.Page(
        '# Simple dashboard of MATSim Simulation Results',
        '## Simulation configuration',
        dp.Table(scenario.scenario_config()),

        '## Total Population Statistics',
        dp.Text('Valid trips = Trips with trip distance > 0'),
        dp.Text(f'Total population = {scenario.mobility_indicators_total("whole_day")[0]}'),
        dp.Table(scenario.mobility_indicators_total("whole_day")[1]),
        dp.Plot(scenario.mobility_indicators_total("whole_day")[2]),

        '## Residents Population Statistics',
        dp.Text('Residents: '
                ' \n - start is in zone Z and start activity type = home or '
                ' \n - end is in zone Z and end activity type = home '),
        dp.Text(f'Residents population = {mobility_indicators_other_whole_day.residents_population}'),
        dp.Table(mobility_indicators_other_whole_day.residents_df),
        dp.Group(blocks=[dp.Plot(mobility_indicators_other_whole_day.residents_regional_graph),
                         dp.Plot(mobility_indicators_other_whole_day.residents_local_graph)], columns=2),

        '## Commuter Statistics',
        dp.Text('Commuters: '
                ' \n - start is in zone Z and start activity type = work or '
                ' \n - end is in zone Z and end activity type = work '
                ' \n - Is not a resident '),
        dp.Text(f'Commuter population = {mobility_indicators_other_whole_day.commuters_population}'),
        dp.Table(mobility_indicators_other_whole_day.commuters_df),
        dp.Group(blocks=[dp.Plot(mobility_indicators_other_whole_day.commuters_regional_graph),
                         dp.Plot(mobility_indicators_other_whole_day.commuters_local_graph)], columns=2),

        '## Visitor Statistics',
        dp.Text('Tourists: '
                ' \n - start or end is in zone Z'
                ' \n - is not a resident or a commuter'),
        dp.Text(f'Visitor population = {mobility_indicators_other_whole_day.tourists_population}'),
        dp.Table(mobility_indicators_other_whole_day.tourists_df),
        dp.Group(blocks=[dp.Plot(mobility_indicators_other_whole_day.tourists_regional_graph),
                         dp.Plot(mobility_indicators_other_whole_day.tourists_local_graph)], columns=2),
            title='Whole Day'
    ),
        dp.Page(
            '## Total Population Statistics',
            dp.Text('Valid trips = Trips with trip distance > 0'),
            dp.Text(f'Total population = {scenario.mobility_indicators_total("PPM")[0]}'),
            dp.Table(scenario.mobility_indicators_total("PPM")[1]),
            dp.Plot(scenario.mobility_indicators_total("PPM")[2]),
            
            '## Residents Population Statistics',
            dp.Text('Residents: '
                    ' \n - start is in zone Z and start activity type = home or '
                    ' \n - end is in zone Z and end activity type = home '),
            dp.Text(f'Residents population = {mobility_indicators_other_PPM.residents_population}'),
            dp.Table(mobility_indicators_other_PPM.residents_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPM.residents_regional_graph),
                             dp.Plot(mobility_indicators_other_PPM.residents_local_graph)], columns=2),

            '## Commuter Statistics',
            dp.Text('Commuters: '
                    ' \n - start is in zone Z and start activity type = work or '
                    ' \n - end is in zone Z and end activity type = work '
                    ' \n - Is not a resident '),
            dp.Text(f'Commuter population = {mobility_indicators_other_PPM.commuters_population}'),
            dp.Table(mobility_indicators_other_PPM.commuters_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPM.commuters_regional_graph),
                             dp.Plot(mobility_indicators_other_PPM.commuters_local_graph)], columns=2),

            '## Visitor Statistics',
            dp.Text('Tourists: '
                    ' \n - start or end is in zone Z'
                    ' \n - is not a resident or a commuter'),
            dp.Text(f'Visitor population = {mobility_indicators_other_PPM.tourists_population}'),
            dp.Table(mobility_indicators_other_PPM.tourists_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPM.tourists_regional_graph),
                             dp.Plot(mobility_indicators_other_PPM.tourists_local_graph)], columns=2),
            title='Morning Peak: 6am - 10am'
        ),
        dp.Page(
            '## Total Population Statistics',
            dp.Text('Valid trips = Trips with trip distance > 0'),
            dp.Text(f'Total population = {scenario.mobility_indicators_total("PCJ")[0]}'),
            dp.Table(scenario.mobility_indicators_total("PCJ")[1]),
            dp.Plot(scenario.mobility_indicators_total("PCJ")[2]),

            '## Residents Population Statistics',
            dp.Text('Residents: '
                    ' \n - start is in zone Z and start activity type = home or '
                    ' \n - end is in zone Z and end activity type = home '),
            dp.Text(f'Residents population = {mobility_indicators_other_PCJ.residents_population}'),
            dp.Table(mobility_indicators_other_PCJ.residents_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PCJ.residents_regional_graph),
                             dp.Plot(mobility_indicators_other_PCJ.residents_local_graph)], columns=2),

            '## Commuter Statistics',
            dp.Text('Commuters: '
                    ' \n - start is in zone Z and start activity type = work or '
                    ' \n - end is in zone Z and end activity type = work '
                    ' \n - Is not a resident '),
            dp.Text(f'Commuter population = {mobility_indicators_other_PCJ.commuters_population}'),
            dp.Table(mobility_indicators_other_PCJ.commuters_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PCJ.commuters_regional_graph),
                             dp.Plot(mobility_indicators_other_PCJ.commuters_local_graph)], columns=2),

            '## Visitor Statistics',
            dp.Text('Tourists: '
                    ' \n - start or end is in zone Z'
                    ' \n - is not a resident or a commuter'),
            dp.Text(f'Visitor population = {mobility_indicators_other_PCJ.tourists_population}'),
            dp.Table(mobility_indicators_other_PCJ.tourists_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PCJ.tourists_regional_graph),
                             dp.Plot(mobility_indicators_other_PCJ.tourists_local_graph)], columns=2),
            title='Off-Peak Period: 10am-4pm'
        ),
        dp.Page(
            '## Total Population Statistics',
            dp.Text('Valid trips = Trips with trip distance > 0'),
            dp.Text(f'Total population = {scenario.mobility_indicators_total("PPS")[0]}'),
            dp.Table(scenario.mobility_indicators_total("PPS")[1]),
            dp.Plot(scenario.mobility_indicators_total("PPS")[2]),

            '## Residents Population Statistics',
            dp.Text('Residents: '
                    ' \n - start is in zone Z and start activity type = home or '
                    ' \n - end is in zone Z and end activity type = home '),
            dp.Text(f'Residents population = {mobility_indicators_other_PPS.residents_population}'),
            dp.Table(mobility_indicators_other_PPS.residents_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPS.residents_regional_graph),
                             dp.Plot(mobility_indicators_other_PPS.residents_local_graph)], columns=2),

            '## Commuter Statistics',
            dp.Text('Commuters: '
                    ' \n - start is in zone Z and start activity type = work or '
                    ' \n - end is in zone Z and end activity type = work '
                    ' \n - Is not a resident '),
            dp.Text(f'Commuter population = {mobility_indicators_other_PPS.commuters_population}'),
            dp.Table(mobility_indicators_other_PPS.commuters_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPS.commuters_regional_graph),
                             dp.Plot(mobility_indicators_other_PPS.commuters_local_graph)], columns=2),

            '## Visitor Statistics',
            dp.Text('Tourists: '
                    ' \n - start or end is in zone Z'
                    ' \n - is not a resident or a commuter'),
            dp.Text(f'Visitor population = {mobility_indicators_other_PPS.tourists_population}'),
            dp.Table(mobility_indicators_other_PPS.tourists_df),
            dp.Group(blocks=[dp.Plot(mobility_indicators_other_PPS.tourists_regional_graph),
                             dp.Plot(mobility_indicators_other_PPS.tourists_local_graph)], columns=2),
            title='Evening Peak Period: 4pm-8pm'
        ),
        dp.Page(
            '# Mobility Indicators - Modal Share',
            dp.Plot(chart1),
            dp.Plot(chart2),
            dp.Plot(chart3),
            '# Presence on the Territory',
            dp.Plot(scenario.plot_presence()),
            '# Average distance per trip motive',
            # dp.Plot(distplot), dp.Plot(timeplot),
            # dp.Plot(scenario.plot_distance_motive()),
            dp.Plot(distplot),
            title='Mobility Indicators'
        )
        )

    name = scenario.path_data.split('/')[-1]
    report.save(f"{name}.html", open=True)


if __name__ == '__main__':
    test = all_outputs()
    dashboard_datapane_matsim(test)
