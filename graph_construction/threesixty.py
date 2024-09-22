import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
from munkres import Munkres
import math
import matplotlib.path as mplPath

def get_visible_area_inside_points(visible_area):
    loop = visible_area
    
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




def make_threesixty(ljdf: pd.DataFrame, return_only_frame: bool = False) -> pd.DataFrame:
    """
    Aggregates the 360 frames for each pass event in the source dataframe.
    :param ljdf: dataframe, containing the 360 frames for each pass event.
    :return: df: dataframe with the 360 frames aggregated.
    """
    loc_col = 'location_y'
    if 'location_y' not in ljdf.columns: loc_col = 'location'
    aggregated_frames_df = ljdf.groupby(['id']).agg({'teammate': lambda x: list(x),
                                                               'actor': lambda x: list(x),
                                                               'keeper': lambda x: list(x),
                                                               loc_col: lambda x: list(x),
                                                               'visible_area': lambda x: min(x)})

    event_dataframes = []
    only_x_coords = []
    only_y_coords = []

    ### dirty way of zipping -- rework in future
    for zipped_tuple in zip(list(aggregated_frames_df['teammate']), list(aggregated_frames_df['actor']),
                            list(aggregated_frames_df['keeper']), list(aggregated_frames_df[loc_col])):
        t = []
        a = []
        k = []
        loc = []
        l_x = []
        l_y = []
        for (is_teammate, is_actor, is_keeper, location) in zip(zipped_tuple[0], zipped_tuple[1], zipped_tuple[2],
                                                                zipped_tuple[3]):
            t.append(is_teammate)
            a.append(is_actor)
            k.append(is_keeper)
            loc.append(location)
            location_x = location[0]
            location_y = location[1]
            l_x.append(location_x)
            l_y.append(location_y)
        df_event = pd.DataFrame(
            {'is_teammate': t, 'is_actor': a, 'is_keeper': k, 'location': loc, 'location_x': l_x, 'location_y': l_y})
        event_dataframes.append(df_event)
        only_x_coords.append(l_x)
        only_y_coords.append(l_y)

    aggregated_frames_df['360_loc_x'] = only_x_coords
    aggregated_frames_df['360_loc_y'] = only_y_coords
    aggregated_frames_df['frames_dfs'] = event_dataframes
    aggregated_frames_df.columns = ['360_frame_teammate', '360_frame_actor', '360_frame_keeper',
                                    '360_frame_player_location', '360_visible_area', '360_location_x', '360_location_y',
                                    '360_frame_df']
    
    inside_points = [tuple(get_visible_area_inside_points(i)) for i in aggregated_frames_df['360_visible_area']]
    aggregated_frames_df['points_in_visible_area'] = inside_points

    if return_only_frame:
        return aggregated_frames_df[['360_frame_df', 'points_in_visible_area']]

    
    return aggregated_frames_df


from scipy.spatial.distance import euclidean
import numpy as np
from munkres import Munkres

def do_munk(pass_df, preceding_df, t, pass_id, duration=1):
    d = {}
    duration = max(1, duration)
    for idx, row_pass in pass_df.iterrows():
        d[idx] = []
        for jdx, row_preceding in preceding_df.iterrows():
            d[idx].append(euclidean(row_pass['location'], row_preceding['location']))
    d['preceding_idx'] = preceding_df.index
    cost_df = pd.DataFrame(d)
    m = Munkres()
    ### If there are more players in the pass_df than in the preceding_df, we need to transpose the cost matrix
    if cost_df.shape[0]>= cost_df.shape[1]:
        indexes = m.compute(np.array(cost_df.iloc[:, :-1].T))
        assignments = [None for i in range(len(cost_df))]
        for pair in indexes:
            assignments[pair[1]] = cost_df.columns[pair[0]]
    else:
        indexes = m.compute(np.array(cost_df.iloc[:, :-1]))
        assignments = [None for i in range(len(cost_df))]
        for pair in indexes:
            assignments[pair[0]] = cost_df.columns[pair[1]]
    cost_df['assignments_from_hungarian'] = pd.to_numeric(pd.Series(assignments), errors='coerce').astype('Int64')
    ass = cost_df[['preceding_idx', 'assignments_from_hungarian']].dropna(subset=['assignments_from_hungarian'])
    n_reset_trajectories = 0
    for (preceding_player, player_in_pass) in zip(ass['preceding_idx'], ass['assignments_from_hungarian']):
        pass_position = pass_df.loc[player_in_pass,'location']
        preceding_position = preceding_df.loc[preceding_player,'location']
        trajectory = [pass_position[0] - preceding_position[0], pass_position[1] - preceding_position[1]]
        trajectory = [i/duration for i in trajectory]
        traj_len = math.sqrt((trajectory[0]**2) + (trajectory[1]**2))
        if traj_len > 13:
            trajectory = [0, 0]
            n_reset_trajectories += 1
        t[player_in_pass] = trajectory
    if n_reset_trajectories >= 3: raise ValueError('Too many reset trajectories.')
    return t

def generate_trajectories(event_series):

    ### process teammates
    duration_from_previous = event_series['sec_difference']
    pass_df = event_series['360_frame_df']
    trajectories = [None] * len(pass_df)
    pass_df_teammates = pass_df[pass_df['is_teammate'] & ~pass_df['is_keeper'] & pass_df['was_point_in_view_before']]
    pass_df_teammates_no_prevfiltering =  pass_df[pass_df['is_teammate'] & ~pass_df['is_keeper']]
    preceding_df = event_series['linked_preceding_event_360_frame']
    preceding_df_teammates = preceding_df[preceding_df['is_teammate'] & ~preceding_df['is_keeper']]

    if len(preceding_df_teammates)>len(pass_df_teammates) and len(pass_df_teammates_no_prevfiltering)>len(pass_df_teammates):
        pass_df_teammates = pass_df_teammates_no_prevfiltering


    try:
        trajectories = do_munk(pass_df_teammates, preceding_df_teammates, trajectories, event_series.id, duration=duration_from_previous)
    except Exception as e:
        event_series['360_frame_df'] = None
        return event_series
        

    ### process opponents
    pass_df_opps = pass_df[~pass_df['is_teammate'] & ~pass_df['is_keeper'] & pass_df['was_point_in_view_before']]
    pass_df_opps_no_prevfiltering =  pass_df[~pass_df['is_teammate'] & ~pass_df['is_keeper']]
    preceding_df_opps = preceding_df[~preceding_df['is_teammate'] & ~preceding_df['is_keeper']]

    if len(preceding_df_opps)>len(pass_df_opps) and len(pass_df_opps_no_prevfiltering)>len(preceding_df_opps):
        pass_df_opps = pass_df_opps_no_prevfiltering

    if len(pass_df_opps) > 0:
        try:
            trajectories = do_munk(pass_df_opps, preceding_df_opps, trajectories, event_series.id, duration=duration_from_previous)
        except Exception as e:
            event_series['360_frame_df'] = None
            return event_series
    ### process keepers
    for is_teammate in [True, False]:
        pass_df_keepers = pass_df[(pass_df['is_keeper']) & (pass_df['is_teammate']==is_teammate)]
        preceding_df_keepers = preceding_df[(preceding_df['is_keeper']) & (preceding_df['is_teammate']==is_teammate)]
        if len(pass_df_keepers) > 0 and len(preceding_df_keepers) > 0:
            keeper_loc = pass_df_keepers['location'].values[0]
            prev_keeper_loc = preceding_df_keepers['location'].values[0]
            trajectory = [keeper_loc[0] - prev_keeper_loc[0], keeper_loc[1] - prev_keeper_loc[1]]
            keeper_index = pass_df_keepers.index.values[0]
            trajectories[keeper_index] = [i/duration_from_previous for i in trajectory]
    
    ### add zeros for players that were not assigned a trajectory --- they were not in frame and we assume stationary.
    for i in range(len(trajectories)):
        if trajectories[i] is None:
            trajectories[i] = [0,0]
    
    pass_df['trajectory'] = trajectories
    pass_df['trajectory_x'] = [x[0] for x in trajectories]
    pass_df['trajectory_y'] = [x[1] for x in trajectories]
    event_series['360_frame_df'] = pass_df
    return event_series




