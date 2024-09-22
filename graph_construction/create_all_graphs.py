from make_pass_master_df import get_pass_master_df
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import dill
from statsbombpy import sb
import matplotlib.path as mplPath
from scipy.spatial.distance import euclidean
import re
from munkres import Munkres
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import math
from torch_geometric.utils import from_networkx
import torch
from pitch_zone import return_pitch_zone_index
from tqdm import tqdm
import scipy
import warnings
from multiprocessing import Pool
import multiprocessing
import functools
warnings.filterwarnings('ignore')


class dummyReturnZero:
    """
    A wrapper function mirroring the implementation of scipy.kde, for when the "evaluate" function needs to return zero
    """
    def evaluate(self, x):
        return 0


### CONSTANTS
with open('./all_positions_kdes_wrapped_updated.dill', 'rb') as f:
    df_all_positions_kde = dill.load(f)

with open('./all_positions_kdes_wrapped_opposition.dill', 'rb') as f:
    df_all_positions_kde_opps = dill.load(f)

def get_integer_surroundings(A, B, X,Y):
    """
    Returns the integer coordinates describing a square, surrounding point (A,B)
    :param A, B: X,Y coordinates of which we want to compute integer surroundings
    :param X: #of points to the left and right of (A,B) to include
    :param Y: #of points to the top and bottom of (A,B) to include
    :return: array of surrounding point coordinates
    """
    return [(i, j) for i in range(A - X, A + X + 1) for j in range(B - Y, B + Y + 1)]

def get_updated_360_df(event_series, player_match_kde, df_all_positions_kde, player_match_kde_opps, df_all_positions_kde_opps):
    """
    Updates the input event_series['360_frame_df'] with the player vectors, based on the Munkres algorithm.
    """
    try:
        assert event_series['player'] in list(event_series['formatted_formation'].player)
    except AssertionError:
        ## This is a pass event where the player is not on the pitch. We don't process these, it is a rare error in processing the substitution events
        return None
    
    #### HANDLE THIS TEAM #######
    gk_name = event_series['formatted_formation'].loc[event_series['formatted_formation'].position == 'Goalkeeper', 'player'].values[0]
    actor_id = event_series['player_id']
    team = event_series['team']
    players_on_pitch = list(event_series['formatted_formation'].player)
    positions_on_pitch = list(event_series['formatted_formation'].position)
    actor_position = event_series['formatted_formation'].loc[event_series['formatted_formation'].player == event_series['player'], 'position'].values[0]
    sample_df = event_series['360_frame_df']
    team_df = dc(sample_df[sample_df['is_teammate'] & ~sample_df['is_actor'] & ~sample_df['is_keeper']])
    if len(team_df) < 3: 
        ### We don't process pass graphs where less than 3 teammates are visible.
        return None
    
    pos_kdes = df_all_positions_kde[~df_all_positions_kde.position.isin(['Goalkeeper', actor_position])][df_all_positions_kde.position.isin(positions_on_pitch)][['position', 'kde']]
    player_kdes = player_match_kde[player_match_kde.player_id != actor_id][player_match_kde.player != gk_name][player_match_kde.team==team][player_match_kde.player.isin(players_on_pitch)][['player', 'kde']]
    if len(player_kdes) != len(pos_kdes): 
        #print('Returning None -- mismatch in number of players and positions')
        return None ### We don't process pass graphs where the number of players and positions don't match -- this is an error in computing substitutions.
    ### We do this to get the most recent positions, as they are not always the same as in the match kde dataframe.
    player_kdes_with_positions = player_kdes.merge(event_series['formatted_formation'], on='player')[['player', 'position', 'kde']]
    pos_joint = pos_kdes.join(player_kdes_with_positions.set_index('position'), on='position', rsuffix='_player')
    pos_joint = pos_joint.dropna(subset=['player', 'kde_player'])
    d_positional = {'position' : pos_joint.position, 'player' : pos_joint.player}
    d_player = {'position' : pos_joint.position, 'player' : pos_joint.player}
    for (index_of_dot, location) in team_df[['location']].iterrows():
        loc = location[0]
        surrounding_locations = get_integer_surroundings(int(loc[0]), int(loc[1]), 3,3)
        try:
            densities_for_dot_surroundings_positional = [sum([kde.evaluate(x) for x in surrounding_locations])[0] for kde in pos_joint['kde']]
            densities_for_dot_surroundings_player =  [sum([kde.evaluate(x) for x in surrounding_locations])[0] if pd.notna(kde) else 0 for kde in pos_joint['kde_player']]
        except TypeError:
            return None
        d_positional.update({f'player_{index_of_dot}' : [density for density in densities_for_dot_surroundings_positional]})
        d_player.update({f'player_{index_of_dot}' : [density for density in densities_for_dot_surroundings_player]})
    ### Scale the data in each DF
    positional_density_df = pd.DataFrame(d_positional)
    transformed_data = MinMaxScaler(feature_range=(1,2)).fit_transform(np.array(positional_density_df.drop(['position', 'player'], axis=1)).reshape(-1, 1))

    positional_density_df.iloc[:,2:] = transformed_data.reshape(np.array(positional_density_df.iloc[:,2:]).shape)
    player_density_df = pd.DataFrame(d_player)
    transformed_data = MinMaxScaler(feature_range=(1,2)).fit_transform(np.array(player_density_df.drop(['position', 'player'], axis=1)).reshape(-1, 1))
    player_density_df.iloc[:,2:] = transformed_data.reshape(np.array(player_density_df.iloc[:,2:]).shape)
    combined_density = {'position' : pos_joint.position, 'player' : pos_joint.player}
    for column in player_density_df.columns:
        if column in ['position', 'player']: continue
        combined_density[column] = [-(0.4*i+0.6*j) for i,j in zip(positional_density_df[column], player_density_df[column])]
    final_cost_df = pd.DataFrame(combined_density)
    m = Munkres()
    indexes = m.compute(np.array(final_cost_df.iloc[:, 2:].T))
    assignments = [None for i in range(len(final_cost_df))]
    for pair in indexes:
        assignments[pair[1]] = final_cost_df.columns[pair[0]+2]

    new_cost_df = dc(final_cost_df)
    new_cost_df['assignments_from_hungarian'] = assignments
    new_cost_df['assignment_index'] = [i.split('_')[-1] if i is not None else None for i in new_cost_df['assignments_from_hungarian']]
    new_cost_df['assignment_index'] = pd.to_numeric(new_cost_df['assignment_index'], errors='coerce').astype('Int64')
    e = new_cost_df[~new_cost_df.assignment_index.isnull()].sort_values(by='assignment_index')[['position', 'player']]
    e.index = new_cost_df[~new_cost_df.assignment_index.isnull()].sort_values(by='assignment_index')['assignment_index']
    team_df[['position', 'player']] = e
    team_df
    concatted_360_df = dc(pd.concat([sample_df, team_df['position'], team_df['player']], axis=1))
    concatted_360_df
    ### Handle actor position
    concatted_360_df['position'][concatted_360_df[concatted_360_df.is_actor]['position'].index[0]] = actor_position
    concatted_360_df['player'][concatted_360_df[concatted_360_df.is_actor]['player'].index[0]] = event_series['player']
    concatted_360_df
    ### Handle GK position
    non_actor_gk = concatted_360_df[(concatted_360_df.is_keeper) & (concatted_360_df.is_teammate) & (~concatted_360_df.is_actor)]
    if len(non_actor_gk)>0:
        ind = non_actor_gk.index[0]
        concatted_360_df['position'][ind] = 'Goalkeeper'
        concatted_360_df['player'][ind] = gk_name
    concatted_360_df = concatted_360_df.merge(event_series['formatted_formation'], on=['position', 'player'], how='left')
    concatted_360_df['player_vector'] = concatted_360_df['player_vector'].apply(lambda x: x[0] if type(x) == list else x)

    #### HANDLE OPPONENT TEAM #######

    gk_name = event_series['formatted_formation_opp'].loc[event_series['formatted_formation_opp'].position == 'Goalkeeper', 'player'].values[0]
    team = event_series['team_opp']
    players_on_pitch = list(event_series['formatted_formation_opp'].player)
    positions_on_pitch = list(event_series['formatted_formation_opp'].position)
    sample_df = event_series['360_frame_df']
    team_df = dc(sample_df[~sample_df['is_teammate'] & ~sample_df['is_keeper']])
    if len(team_df) < 3: 
        ### We don't process pass graphs where less than 3 teammates are visible.
        return None
    pos_kdes = df_all_positions_kde_opps[~df_all_positions_kde_opps.position.isin(['Goalkeeper'])][df_all_positions_kde_opps.position.isin(positions_on_pitch)][['position', 'kde']]
    player_kdes = player_match_kde_opps[player_match_kde_opps.player != gk_name][player_match_kde_opps.team==team][player_match_kde_opps.player.isin(players_on_pitch)][['player', 'kde']]
    if len(player_kdes) != len(pos_kdes): 
        #print('Returning None -- mismatch in number of players and positions')
        return None ### We don't process pass graphs where the number of players and positions don't match -- this is an error in computing substitutions.
    ### We do this to get the most recent positions, as they are not always the same as in the match kde dataframe.
    player_kdes_with_positions = player_kdes.merge(event_series['formatted_formation_opp'], on='player')[['player', 'position', 'kde']]
    pos_joint = pos_kdes.join(player_kdes_with_positions.set_index('position'), on='position', rsuffix='_player')
    pos_joint = pos_joint.dropna(subset=['player', 'kde_player'])
    d_positional = {'position' : pos_joint.position, 'player' : pos_joint.player}
    d_player = {'position' : pos_joint.position, 'player' : pos_joint.player}
    for (index_of_dot, location) in team_df[['location']].iterrows():
        loc = location[0]
        surrounding_locations = get_integer_surroundings(int(loc[0]), int(loc[1]), 3,3)
        try:
            densities_for_dot_surroundings_positional = [sum([kde.evaluate(x) for x in surrounding_locations])[0] for kde in pos_joint['kde']]
            densities_for_dot_surroundings_player =  [sum([kde.evaluate(x) for x in surrounding_locations])[0] if pd.notna(kde) else 0 for kde in pos_joint['kde_player']]
        except TypeError:
            return None
        d_positional.update({f'player_{index_of_dot}' : [density for density in densities_for_dot_surroundings_positional]})
        d_player.update({f'player_{index_of_dot}' : [density for density in densities_for_dot_surroundings_player]})
    ### Scale the data in each DF
    positional_density_df = pd.DataFrame(d_positional)
    transformed_data = MinMaxScaler(feature_range=(1,2)).fit_transform(np.array(positional_density_df.drop(['position', 'player'], axis=1)).reshape(-1, 1))
    
    positional_density_df.iloc[:,2:] = transformed_data.reshape(np.array(positional_density_df.iloc[:,2:]).shape)
    player_density_df = pd.DataFrame(d_player)
    transformed_data = MinMaxScaler(feature_range=(1,2)).fit_transform(np.array(player_density_df.drop(['position', 'player'], axis=1)).reshape(-1, 1))
    player_density_df.iloc[:,2:] = transformed_data.reshape(np.array(player_density_df.iloc[:,2:]).shape)
    combined_density = {'position' : pos_joint.position, 'player' : pos_joint.player}
    for column in player_density_df.columns:
        if column in ['position', 'player']: continue
        combined_density[column] = [-(0.4*i+0.6*j) for i,j in zip(positional_density_df[column], player_density_df[column])]
    final_cost_df = pd.DataFrame(combined_density)
    m = Munkres()
    indexes = m.compute(np.array(final_cost_df.iloc[:, 2:].T))
    assignments = [None for i in range(len(final_cost_df))]
    for pair in indexes:
        assignments[pair[1]] = final_cost_df.columns[pair[0]+2]

    new_cost_df = dc(final_cost_df)
    new_cost_df['assignments_from_hungarian'] = assignments
    new_cost_df['assignment_index'] = [i.split('_')[-1] if i is not None else None for i in new_cost_df['assignments_from_hungarian']]
    new_cost_df['assignment_index'] = pd.to_numeric(new_cost_df['assignment_index'], errors='coerce').astype('Int64')
    e = new_cost_df[~new_cost_df.assignment_index.isnull()].sort_values(by='assignment_index')[['position', 'player']]
    e.index = new_cost_df[~new_cost_df.assignment_index.isnull()].sort_values(by='assignment_index')['assignment_index']
    team_df[['position', 'player']] = e
    concatted_360_df.loc[~concatted_360_df.is_teammate & ~concatted_360_df.is_keeper, ['position', 'player']] = team_df[['position', 'player']]
    for idx in concatted_360_df.loc[~concatted_360_df.is_teammate & ~concatted_360_df.is_keeper].index:
        ply = concatted_360_df['player'][idx]
        filtered_formation = event_series['formatted_formation_opp'].loc[event_series['formatted_formation_opp'].player == ply]
        pv = filtered_formation['player_vector'].values[0]
        concatted_360_df['player_vector'][idx] = pv
    non_actor_gk = concatted_360_df[(concatted_360_df.is_keeper) & (~concatted_360_df.is_teammate)]
    if len(non_actor_gk)>0:
        ind = non_actor_gk.index[0]
        concatted_360_df['position'][ind] = 'Goalkeeper'
        concatted_360_df['player'][ind] = gk_name
        keeper_vector =  event_series['formatted_formation_opp'].loc[event_series['formatted_formation_opp'].player == gk_name].values[0]
        concatted_360_df['player_vector'][ind] = keeper_vector
    concatted_360_df['player_vector'] = concatted_360_df['player_vector'].apply(lambda x: x[0] if type(x) == list else x)
    concatted_360_df





    return concatted_360_df


def get_match_kdes(mid):
    """
    Returns an dataframe of player location KDE distributions for the observed match 'mid' for the actor's teammates
    """

    events = sb.events(match_id = mid, split=False, flatten_attrs=True)
    all_on_ball_events = events[~pd.isna(events.player_id) & ~pd.isna(events.location)&(events['play_pattern'].isin(['Regular Play']))]
    all_on_ball_events = all_on_ball_events[['team', 'possession_team_id', 'player', 'position', 'player_id', 'location']]
    all_on_ball_events.loc[all_on_ball_events.position != 'Goalkeeper']
    d2 = {'team' : [], 'position' : [], 'player_id' : [], 'player' : [], 'locs' : [], 'kde' : []}

    for team in all_on_ball_events['team'].unique():
        all_team_events = all_on_ball_events[all_on_ball_events['team'] == team]
        for (player, player_id) in zip(all_team_events['player'].unique(), all_team_events['player_id'].unique()):
            all_player_events = all_team_events[all_team_events.player_id == player_id]
            l_x = [i[0] for i in all_player_events['location']]
            l_y = [i[1] for i in all_player_events['location']]
            pos = [i for i in all_player_events['position']]            
            d2['team'].append(team)
            d2['position'].append(pos[0])
            d2['player_id'].append(player_id)
            d2['player'].append(player)
            loc_df = pd.DataFrame({'l_x' : l_x, 'l_y' : l_y})
            d2['locs'].append(loc_df)
            ## if the player has less than 3 distinct on-ball events, don't do KDE for them
            if len(loc_df)  <= 3:
                d2['kde'].append(dummyReturnZero())
                continue
            kde = gaussian_kde(np.array(loc_df).T, bw_method=1)
            if pd.isna(kde): kde = dummyReturnZero()
            assert kde is not None
            assert pd.notna(kde)
            d2['kde'].append(kde)
    return pd.DataFrame(d2)

def get_opp_match_kdes(mid):
    """
    Returns an dataframe of player location KDE distributions for the observed match 'mid' for the actor's opponents
    """
    events = sb.events(match_id = mid, split=False, flatten_attrs=True)
    all_on_ball_events = events[~pd.isna(events.player_id) & ~pd.isna(events.location)&(events['play_pattern'].isin(['Regular Play']))]
    all_on_ball_events = all_on_ball_events[['team', 'possession_team_id', 'player', 'position', 'player_id', 'location']]
    all_on_ball_events.loc[all_on_ball_events.position != 'Goalkeeper']
    d2 = {'team' : [], 'position' : [], 'player_id' : [], 'player' : [], 'locs' : [], 'kde' : []}

    for team in all_on_ball_events['team'].unique():
        all_team_events = all_on_ball_events[all_on_ball_events['team'] == team]
        for (player, player_id) in zip(all_team_events['player'].unique(), all_team_events['player_id'].unique()):
            all_player_events = all_team_events[all_team_events.player_id == player_id]
            l_x = [120-i[0] for i in all_player_events['location']]
            l_y = [80-i[1] for i in all_player_events['location']]
            pos = [i for i in all_player_events['position']]            
            d2['team'].append(team)
            d2['position'].append(pos[0])
            d2['player_id'].append(player_id)
            d2['player'].append(player)
            loc_df = pd.DataFrame({'l_x' : l_x, 'l_y' : l_y})
            d2['locs'].append(loc_df)
            ## if the player has less than 3 distinct on-ball events, don't do KDE for them
            if len(loc_df)  <= 3:
                d2['kde'].append(dummyReturnZero())
                continue
            kde = gaussian_kde(np.array(loc_df).T, bw_method=1)
            if pd.isna(kde): kde = dummyReturnZero()
            assert kde is not None
            assert pd.notna(kde)
            d2['kde'].append(kde)
    return pd.DataFrame(d2)

def get_inside_points(ev):
    """
    For the observed row 'ev', get all the integer points inside the 360_visible_area polygon.
    """
    loop = ev['360_visible_area']
    
    # Convert the loop to a numpy array of shape (n, 2)
    loop = np.array(loop).reshape(-1, 2)

    # Create a Path object from the loop
    path = mplPath.Path(loop)

    # Define an array of points to test
    points = np.array([[x, y] for x in range(0, 120) for y in range(0, 80)])

    # Use the contains_points method to get a boolean array
    inside = path.contains_points(points)

    # Filter out the points that are outside the loop
    points_inside = points[inside]

    # Print the list of points inside the loop
    inside_points = points_inside.tolist()
    return inside_points

def evaluate_pitch_control_function(point, ev, beta=2.5, eps=1e-8, reaction_time=0.7, player_yps=5.4):
    """
    Evaluate the Spearman pitch control function at the point 'point'.
    """
    event_df = ev['360_frame_df']
    numerator = 0
    denominator = 0
    for (teammate_status, location, trajectory) in zip(event_df['is_teammate'],event_df['location'], event_df['trajectory']):
        #0.7 seconds as reaction time is a constant 
        location_after_reaction = (location[0] + reaction_time*trajectory[0], location[1] + reaction_time*trajectory[1])
        time_to_reach = euclidean(location_after_reaction, point)/player_yps
        if time_to_reach < eps: time_to_reach = eps
        denominator += time_to_reach**(-beta)
        if teammate_status: index = 1 
        else: index = -1
        numerator += (index)*(time_to_reach**(-beta))
    if denominator == 0: denominator += eps
    return (numerator/denominator + 1) / 2

def is_nn_teammate(point, ev):
    """
    returns whether the nearest neighbor to the point 'point' is a teammate, used in Voronoi pitch control evaluation.
    """
    event_df = ev['360_frame_df']
    min_dist = np.inf
    min_state = None
    for (teammate, location) in zip(event_df['is_teammate'], event_df['location']):
        dist = euclidean(point, location)
        if dist < min_dist:
            min_dist = dist
            min_state = teammate
    return min_state

def get_control_df(ev):
    """
    Prepares the control df, used for visualization of pitch control
    """
    try:
        inside_points = get_inside_points(ev)
    except ValueError:
        return None
    control = {'p_x' : [], 'p_y' : [], 'p_full' : [], 'is_controlled_voronoi' :  [], 'control_pcf' : []}
    for point in inside_points:
        control['p_x'].append(point[0])
        control['p_y'].append(point[1])
        control['p_full'].append((point[0], point[1]))
        control['is_controlled_voronoi'].append(is_nn_teammate(point, ev))
        control['control_pcf'].append(evaluate_pitch_control_function(point, ev))
    return pd.DataFrame(control)


def get_fraction_of_controlled_points(point_list, lookup_df):
    """
    Returns the fraction of points that are controlled by teammates, used for Voronoi pitch control.
    """
    if (len(lookup_df.loc[lookup_df.p_full.isin(point_list)]) < 4):
        #print('returning 1 due to small distance pass.')
        return 1
    return len(lookup_df.loc[(lookup_df.p_full.isin(point_list)) & (lookup_df.is_controlled_voronoi)]) / len(lookup_df.loc[lookup_df.p_full.isin(point_list)])

def get_sum_of_control_on_path(point_list, lookup_df):
    """
    Returns the sum of total control on the path, used for Spearman pitch control
    """
    if (len(lookup_df.loc[lookup_df.p_full.isin(point_list)]) < 4):
        #print('returning 1 due to small distance pass.')
        return 1
    return sum(lookup_df.loc[lookup_df.p_full.isin(point_list)]['control_pcf'].values) /  len(lookup_df.loc[lookup_df.p_full.isin(point_list)])

def get_points_on_path(p1, p2, num_points): 
    """
    Returns all integer points on the path between two points (that are within the visible area). If line is long, gets at most num_points.
    """
    p1 = (p1[0], 80-p1[1])
    p2 = (p2[0], 80-p2[1])
    points = []
    
    if p1[0] == p2[0]:  # Vertical path
        y_coords = np.linspace(min(p1[1], p2[1]), max(p1[1], p2[1]), num_points)
        x_coord = p1[0]
        for y in y_coords:
            points.append((x_coord, int(80 - y)))
    min_x = int(min(p1[0], p2[0]))
    max_x = int(max(p1[0], p2[0]))
    smaller = min([p1, p2], key=lambda x:x[0])
    larger = max([p1,p2], key=lambda x:x[0])
    x_coords = np.linspace(min_x, max_x, num_points)
    y_coords = np.interp(x_coords, [smaller[0], larger[0]], [smaller[1], larger[1]])
    for i in range(num_points):
        x = int(x_coords[i])
        y = int(80-y_coords[i])
        if (x, y) not in points:
            points.append((x, y))
    return points

def check_if_valid_point(point):
    """
    Validation to see if point lies within the pitch area.
    """
    if 0 <= point[0] <=120 and 0 <= point[1] <= 80:
        return True
    return False

def get_points_in_circle(point, r):
    """
    Return integer points within the visible area, centered around point (point) with radius R.
    """
    x = int(point[0])
    y = int(point[1])
    points = []
    for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
            if math.sqrt((i - x) ** 2 + (j - y) ** 2) <= r:
                if check_if_valid_point((i,j)):
                    points.append((i, j))
    return points

def get_voronoi_homogeneity_indicator(points, control_df):
    """
    Evaluates whether the points in 'points' have homogeneous transition according to the voronoi pitch control model.
    """
    c = []
    for point in points:
        if point in list(control_df.p_full):
            control = control_df.loc[control_df.p_full == point, 'is_controlled_voronoi'].values[0]
            c.append(control)
    if len(c) < 4: 
        return 1
    number_of_control_transitions_on_line = pd.Series(c).shift().bfill().ne(pd.Series(c)).sum()
    if number_of_control_transitions_on_line > 1:
        is_homogeneous_transition = 0
    else:
        is_homogeneous_transition = 1
    return is_homogeneous_transition

def get_pcf_homogeneity_indicator(points, control_df):
    """
    Evaluates whether the points in 'points' have homogeneous transition according to the spearman pitch control model.
    """
    c = []
    for point in points:
        if point in list(control_df.p_full):
            control = round(control_df.loc[control_df.p_full == point, 'control_pcf'].values[0])
            c.append(control)
    if len(c) < 4: 
        return 1
    number_of_control_transitions_on_line = pd.Series(c).shift().bfill().ne(pd.Series(c)).sum()
    if number_of_control_transitions_on_line > 1:
        is_homogeneous_transition = 0
    else:
        is_homogeneous_transition = 1
    return is_homogeneous_transition

def make_weighted_graph(event_series, positional_kde=None, positional_header=None, match_kde=None, positional_kde_opps=None, match_kde_opps=None, include_player_vectors=True):
    """
    Generates all three NX graph configurations, their PYG Data equivalents, and outputs the associated pitch control DF, used for visualization. If include_player_vectors=True, graphs are also annotated with player vectors from FIFA dataset (note: this may exclude some additional players.)
    """
    
    mid = event_series['match_id']
    control_df = get_control_df(event_series)
    if control_df is None: return None, None, None, None, None, None, None
    if include_player_vectors:
        if match_kde is None: 
            match_kde = get_match_kdes(int(mid))
            match_kde_opps = get_opp_match_kdes(int(mid))
        updated_360_df = get_updated_360_df(event_series, match_kde, positional_kde, match_kde_opps, positional_kde_opps)
    else: updated_360_df = event_series['360_frame_df']
    if updated_360_df is None: 
        return None, None, None, None, None, None, None

    G_fc_with_opps = nx.Graph()
    G_fc_no_opps = nx.Graph()
    G_hs = nx.Graph()

    ### Add nodes
    actor_index = updated_360_df.loc[updated_360_df.is_actor].index[0]
    team = updated_360_df.loc[updated_360_df.is_teammate]
    teammates_no_actor = updated_360_df.loc[(updated_360_df.is_teammate) & (~updated_360_df.is_actor)]

    G_fc_with_opps.add_nodes_from([i for i in updated_360_df.iterrows()])
    G_fc_no_opps.add_nodes_from([i for i in team.iterrows()])
    G_hs.add_nodes_from([i for i in team.iterrows()])

    


    ### Add Edges
    G_fc_no_opps.add_edges_from([(i,j) for i in team.index for j in team.index if i < j])
    G_fc_with_opps.add_edges_from([(i,j) for i in updated_360_df.index for j in updated_360_df.index if i < j])
    G_hs.add_edges_from([(actor_index, i) for i in teammates_no_actor.index])
    
    
    for this_G, has_oops in zip([G_fc_no_opps, G_fc_with_opps, G_hs], [False, True, False]):
        ### Add edge features
        distances = {}
        voronoi_pitch_control = {}
        pcf_pitch_control = {}
        voronoi_transition_homogeneity = {}
        pcf_transition_homogeneity = {}
        has_opps = {}
        for (i, j) in this_G.edges():
            distances[(i,j)] = euclidean(this_G.nodes[i]['location'], this_G.nodes[j]['location'])
            points_on_path = get_points_on_path(this_G.nodes[i]['location'], this_G.nodes[j]['location'], 30)
            voronoi_pitch_control[(i,j)] = get_fraction_of_controlled_points(points_on_path, control_df)
            pcf_pitch_control[(i,j)] = get_sum_of_control_on_path(points_on_path, control_df)
            voronoi_transition_homogeneity[(i,j)] = get_voronoi_homogeneity_indicator(points_on_path, control_df)
            pcf_transition_homogeneity[(i,j)] = get_pcf_homogeneity_indicator(points_on_path, control_df)
            has_opps[(i,j)] = this_G.nodes[i]['is_teammate'] == this_G.nodes[j]['is_teammate']
        nx.set_edge_attributes(this_G, distances, 'distance')
        nx.set_edge_attributes(this_G, voronoi_pitch_control, 'voronoi_pitch_control')
        nx.set_edge_attributes(this_G, pcf_pitch_control, 'pcf_pitch_control')
        nx.set_edge_attributes(this_G, voronoi_transition_homogeneity, 'voronoi_transition_homogeneity')
        nx.set_edge_attributes(this_G, pcf_transition_homogeneity, 'pcf_transition_homogeneity')
        if has_oops:
            nx.set_edge_attributes(this_G, has_opps, 'is_teammate')

        ### Add node features
        voronoi_ps_control_small = {}
        voronoi_ps_control_large = {}
        pcf_ps_control_small = {}
        pcf_ps_control_large = {}
        for i in this_G.nodes():
            points_in_circle_small = get_points_in_circle(this_G.nodes[i]['location'], 4)
            points_in_circle_large = get_points_in_circle(this_G.nodes[i]['location'], 8)
            voronoi_ps_control_small[i] = get_fraction_of_controlled_points(points_in_circle_small, control_df)
            voronoi_ps_control_large[i] = get_fraction_of_controlled_points(points_in_circle_large, control_df)
            pcf_ps_control_small[i] = get_sum_of_control_on_path(points_in_circle_small, control_df)
            pcf_ps_control_large[i] = get_sum_of_control_on_path(points_in_circle_large, control_df)
        nx.set_node_attributes(this_G, voronoi_ps_control_small, 'voronoi_ps_control_small')
        nx.set_node_attributes(this_G, voronoi_ps_control_large, 'voronoi_ps_control_large')
        nx.set_node_attributes(this_G, pcf_ps_control_small, 'pcf_ps_control_small')
        nx.set_node_attributes(this_G, pcf_ps_control_large, 'pcf_ps_control_large')

    try:
        pygg_fc_with_opps = from_networkx(G_fc_with_opps)
        pygg_fc_no_opps = from_networkx(G_fc_no_opps)
        pygg_hs = from_networkx(G_hs)
    except Exception as e:
        return None, None, None, None, None, None, None
    
    for this_pyg_graph, work_with_player_vectors in zip([pygg_fc_with_opps, pygg_fc_no_opps, pygg_hs], [True, True, True]):
        location = torch.zeros(20)
        location_index = return_pitch_zone_index(event_series['pass_end_location'])
        assert location_index is not None
        location[location_index] = 1.
        this_pyg_graph['end_location_zone'] = location
        this_pyg_graph['end_location_coords'] = torch.tensor(event_series['pass_end_location'])
        if work_with_player_vectors and include_player_vectors:
            try:
                this_pyg_graph['player_vector'] = this_pyg_graph['player_vector'].reshape(this_pyg_graph.is_teammate.shape[0],-1)
            except Exception as e:
                under_construction_tensor = []
                for i in this_pyg_graph['player_vector']:
                    try:
                        if type(i) != list: reshaped = i.reshape(1,-1)
                        else: reshaped = i[0].reshape(1,-1)
                    except Exception as e:
                        raise ValueError('Malformed Player Vector')
                    if reshaped.shape[1] != 9:
                        raise ValueError('Malformed Player Vector')
                    else: under_construction_tensor.append(reshaped)
                this_pyg_graph['player_vector'] = torch.tensor(np.stack( under_construction_tensor, axis=0 ).reshape(this_pyg_graph.is_teammate.shape[0],-1))
        this_pyg_graph.pass_angle = torch.tensor(event_series['pass_angle'])
        this_pyg_graph.pass_length = torch.tensor(event_series['pass_length'])
        this_pyg_graph.is_pass_rightfooted = torch.tensor([1 if event_series['pass_body_part'] == 'Right Foot' else 0])
        mask = torch.zeros(len(positional_header))
        for idx, position in enumerate(positional_header):
            if position in list(event_series['formatted_formation']['position']): mask[idx] = 1.
        this_pyg_graph.position_mask = mask
        recipient_vector = torch.zeros(len(positional_header))
        try:
            recipient_position = event_series['formatted_formation'].loc[event_series['formatted_formation'].player == event_series['pass_recipient'], 'position'].values[0]
        except Exception as e:
            return None, None, None, None, None, None, None
        recipient_vector[positional_header.index(recipient_position)] = 1.
        this_pyg_graph.recipient_position_vector = recipient_vector
    return G_fc_no_opps, G_fc_with_opps, G_hs, pygg_fc_no_opps, pygg_fc_with_opps, pygg_hs, control_df

    


    



def construct_all_match_graphs(mid, sid, include_player_vectors=False):
    """
    Main loop for graph construction.
    """
    try:
        pass_master_df = get_pass_master_df(mid, season_id=sid, include_player_vectors=include_player_vectors)
    except Exception as e:
        print('************')
        print(f'Logged exception in constructing pass master df for mid= {mid}. Skipping match.')
        print(e)
        print('************')
        print('\n \n')
        return (mid, None)
    
    if pass_master_df is None:
        print('Skipping match', mid, 'due to lack of data on players.')
        return (mid, None)
    
    all_passes = len(pass_master_df)
    if include_player_vectors:
        try:
            match_kde = get_match_kdes(int(mid))
            macth_kde_opps = get_opp_match_kdes(int(mid))
        except Exception as e:
            print('************')
            print(f'Logged exception in calculating match KDEs for mid= {mid}. Skipping match.')
            print(e)
            print('************')
            print('\n \n')
            return (mid, None)
    positional_kde = df_all_positions_kde
    positional_kde_opps = df_all_positions_kde_opps
    positional_header = list(positional_kde['position'])
    graphs_fc_no_opps = []
    graphs_fc_with_opps = []
    graphs_hs = []

    pyg_data_fc_no_opps = []
    pyg_data_with_opps = []
    pyg_data_hs = []

    control_dfs = []
    for (idx, event_series) in tqdm(pass_master_df.iterrows(), total=all_passes, position=multiprocessing.current_process()._identity[0]):
    #for (idx, event_series) in tqdm(pass_master_df.iterrows(), total=all_passes):
        try:
            if include_player_vectors:
                G_fc_no_opps, G_fc_with_opps, G_hs, pygg_fc_no_opps, pygg_fc_with_opps, pygg_hs, control_df = make_weighted_graph(event_series, positional_kde, positional_header, match_kde, positional_kde_opps, macth_kde_opps, include_player_vectors=True)
            else: 
                G_fc_no_opps, G_fc_with_opps, G_hs, pygg_fc_no_opps, pygg_fc_with_opps, pygg_hs, control_df = make_weighted_graph(event_series, positional_header=positional_header, include_player_vectors=False)
        except Exception as e:
            G_fc_no_opps, G_fc_with_opps, G_hs, pygg_fc_no_opps, pygg_fc_with_opps, pygg_hs, control_df = None, None, None, None, None, None, None
        graphs_fc_no_opps.append(G_fc_no_opps)
        graphs_fc_with_opps.append(G_fc_with_opps)
        graphs_hs.append(G_hs)
        pyg_data_fc_no_opps.append(pygg_fc_no_opps)
        pyg_data_with_opps.append(pygg_fc_with_opps)
        pyg_data_hs.append(pygg_hs)
        control_dfs.append(control_df)
    pass_master_df['graphs_fc_no_opps'] = graphs_fc_no_opps
    pass_master_df['graphs_fc_with_opps'] = graphs_fc_with_opps
    pass_master_df['graphs_hs'] = graphs_hs
    pass_master_df['pyg_data_fc_no_opps'] = pyg_data_fc_no_opps
    pass_master_df['pyg_data_with_opps'] = pyg_data_with_opps
    pass_master_df['pyg_data_hs'] = pyg_data_hs
    pass_master_df = pd.DataFrame(pass_master_df)
    pass_master_df['control_df'] = control_dfs
    pass_master_df.dropna(subset=['graphs_fc_no_opps'], inplace=True)
    return (mid, pass_master_df)







if __name__ == '__main__':
    ac = sb.competitions()
    relevant_competitions = ac.loc[(~ac.match_available_360.isna()) & (ac.competition_gender == 'male') & (ac.country_name.isin(['International', 'Europe']))]
    master_dfs = {'mid' : [], 'master_df' : []}
    for (idx, row) in relevant_competitions.iterrows():
        cid = row['competition_id']
        sid = row['season_id']
        matches = sb.matches(cid, sid)
        pool = Pool(14)
        sids = [sid] * len(matches['match_id'])
        for (mid, match_graphs_df) in pool.imap_unordered(functools.partial(construct_all_match_graphs, sid=sid, include_player_vectors=False), matches['match_id']):
            master_dfs['master_df'].append(match_graphs_df)
            master_dfs['mid'].append(mid)
    master_dfs = pd.DataFrame(master_dfs)
    master_dfs.dropna(subset=['master_df'], inplace=True)
    with open('master_dataframe.dill', 'wb') as f:
        dill.dump(master_dfs, f)
    
    positional_header = list(df_all_positions_kde['position'])
    with open('positional_header.dill', 'wb') as f:
        dill.dump(positional_header, f)   
    
