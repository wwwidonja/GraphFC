from statsbombpy import sb
import pandas as pd
from copy import deepcopy as dc
from player_vectors import get_player_vector
from random import sample
from threesixty import make_threesixty, generate_trajectories 
import re
import dill
from scipy.spatial.distance import euclidean
import numpy as np

def calculate_centroid(coords):
    """Calculate the centroid (mean) of a list of coordinates."""
    x_coords = [x for x, y in coords]
    y_coords = [y for x, y in coords]
    return np.mean(x_coords), np.mean(y_coords)

def detect_zero_indices(lst) -> list:
    zero_indices = [i for i, item in enumerate(lst) if isinstance(item, int) and item == 0]
    return zero_indices

def get_formation_df(tactics, team, fifa_df, include_player_vectors=True) -> pd.DataFrame:
    if include_player_vectors:
        form = {'position' : [], 'player' : [], 'player_vector' : []}
    else: form = {'position' : [], 'player' : []}
    ### For each element in the lineup column, extract the player name and position. Then, get the player vector from the fifa DF. If the player can not be found,  a 0 is returned.
    for lineup_element in tactics['lineup']:
        player_name = lineup_element['player']['name']
        form['player'].append(player_name)
        form['position'].append(lineup_element['position']['name'])
        if include_player_vectors:
            form['player_vector'].append(get_player_vector(team, player_name, fifa_df=fifa_df))
    ## Fetch the indices of players who could not have been found
    if include_player_vectors:
        zero_indices = detect_zero_indices(form['player_vector'])
        ## In case there were players that were not found, pretend that this obscure player is similar to one of their teammates. 
        if len(zero_indices)>7: raise ValueError(f'Too many players not found in match for {team}: \n{[form["player"][i] for i in zero_indices]}')
        if len(zero_indices)>0:
            non_zeros = list(filter(lambda x: type(x) != int, form['player_vector']))
            for index in zero_indices:
                rep = sample(non_zeros, k=1)
                form['player_vector'][index] = rep
    return pd.DataFrame(form)

def get_new_formation(formation_df, sub, fifa_df, include_player_vectors=True) -> pd.DataFrame:
    formation_df_new = dc(formation_df)
    row_index = formation_df_new[formation_df_new.position==sub.position].index[0]
    formation_df_new.player[row_index] = sub.substitution_replacement
    if include_player_vectors:
        new_player_vector = get_player_vector(sub.team, sub.substitution_replacement, fifa_df)
        if type(new_player_vector) == int:
            non_zeros = list(filter(lambda x: type(x) != int, formation_df_new['player_vector']))
            new_player_vector = sample(non_zeros, k=1)
        formation_df_new.player_vector[row_index] = new_player_vector
    return formation_df_new


def get_formations_for_all_minutes(match_id: int, subs: pd.DataFrame, events: pd.DataFrame, sid = None, fifa_df = None, include_player_vectors=True) -> pd.DataFrame:
    """

    :param match_id: statsbomb match id
    :param subs: the dataframe denoting all substitutions. Retrieved by sb.events(match_id, split=True, flatten_attrs=True)['substitutions']
    :param events: the dataframe of all events
    :param sid: the season ID
    :param fifa_df: the Fifa Dataframe to be used
    :return:
    """
    #if not fifa_df and not sid: print('Error! No fifa DF provided. Plase provide at least the season')
    if fifa_df is None:
        if sid==43: n='fifa_20_pca'
        elif sid==106: n='fifa_22_pca'
        elif sid==282: n='fifa_24_pca'
        else: print('Invalid SID!')
        fifa_df = pd.read_csv(f'./fifa_data/{n}.csv')

    max_minute = max(events['passes'].minute)

    ### create reference dataframe with the starting XIs and the tactical changes
    starting_xis = dc(events['starting_xis'][['team', 'minute', 'tactics']])
    starting_xis.minute = [0,0]
    ref = pd.concat([starting_xis, events['tactical_shifts'][['team', 'minute', 'tactics']]], axis=0).reset_index(drop=True)
   
    try:
        ref['formatted_formation'] = [get_formation_df(tactic, team, fifa_df, include_player_vectors=include_player_vectors) for tactic, team in zip(ref.tactics, ref.team)]
    except Exception as e:
        print(e)
        return None
    ref = ref.drop('tactics', axis=1)
    ref = ref.sort_values(by='minute')

    master_frame = []
    for team in subs.team.unique():

        ##create a sub-frame for each team
        subs_team = subs[subs.team == team].reset_index(drop=True)

        ### Check if we will be adjusting for substitutions in for this team
        if len(subs_team) > 0:
            still_looking_for_subs = True
        else:
            still_looking_for_subs = False


        sub_idx = 0
        next_sub_minute = subs_team.minute[sub_idx]
        rolling_sample_row = None

        ## for each minute in the game
        for minute in range(max_minute + 1):
            event_happened = False
            minute_row = ref[(ref.team == team) & (ref.minute == minute)]
            ### check if it's in the reference dataframe (a tactical shift has happened, or a starting XI is provided)
            if len(minute_row) >= 1:
                event_happened = True
                rolling_sample_row = dc(minute_row).reset_index(drop=True)
                if still_looking_for_subs and minute == next_sub_minute:
                    sub_idx += 1
                    if len(subs_team) == sub_idx:
                        still_looking_for_subs = False
                    else:
                        next_sub_minute = subs_team.minute[sub_idx]

            ### or if a substitution has happened in this minute
            elif minute == next_sub_minute:
                ### if yes, replace the original player in a given position in the formation with ther sub

                event_happened = True
                while (minute == next_sub_minute and still_looking_for_subs):
                    new_formation = get_new_formation(rolling_sample_row.formatted_formation.reset_index(drop=True)[0],
                                                      subs_team.iloc[sub_idx, :], fifa_df, include_player_vectors=include_player_vectors)
                    rolling_sample_row = pd.DataFrame(
                        {'team': [team], 'minute': [minute], 'formatted_formation': [new_formation]})
                    sub_idx += 1
                    if len(subs_team) == sub_idx:
                        still_looking_for_subs = False
                    else:
                        next_sub_minute = subs_team.minute[sub_idx]

            ### if not, just add a copy of the rolling sample with the observed minute and...
            rolling_sample_row.minute = minute
            if len(master_frame) == 0:
                master_frame = dc(rolling_sample_row)
            else: ### concatenate it to the master df being constructed.
                master_frame = pd.concat([master_frame, rolling_sample_row], axis=0).reset_index(drop=True)

    return master_frame

def flip_teams(team, possible_teams):
    if team == possible_teams[0]: return possible_teams[1]
    else: return possible_teams[0]


def make_list_dimension_hashable(df: pd.DataFrame, dimension_name: str) -> pd.DataFrame:
    """
    Reformats list-values in a specific column in source dataframe to tuples, so that the column becomes hashable.
    If the column value is not a list, it leaves it as is.
    :param df: dataframe, containing the column `dimension_name`
    :param dimension_name: string column name
    :return: df: dataframe with the reformatted column dimension_name
    """

    df[dimension_name] = [tuple(i) if type(i) == list else i for i in df[dimension_name]]
    return df

def get_pass_master_df(match_id: int, season_id=None, fifa_df=None, include_player_vectors=True) -> pd.DataFrame:
    """

    :param match_id: id of the match for statsbombpy.sb.events() and frames() function.
    :return: dataframe of passes for which 360 frames are available. 360 frames are aggregated and joined
    onto pass events.
    """
    events = sb.events(match_id, split=True, flatten_attrs=True)
    pass_events_df = events['passes']
    all_events_df = sb.events(match_id, split=False, flatten_attrs=True)
    all_frames_df = sb.frames(match_id)  # get df for frames of the same match

    pass_events_df = pass_events_df[~pd.isna(pass_events_df.player_id) & ~pd.isna(pass_events_df.location) & (~pass_events_df.pass_outcome.isin(['Incomplete', 'Out', 'Pass Offside', 'Unknown', 'Injury Clearance'])
                                                                                                              &(pass_events_df.pass_body_part.isin(['Right Foot', 'Left Foot']))
                                                                                                              &(pass_events_df.pass_length<=40))].reset_index(drop=True) ##Remove this one!

    lj_df = pd.merge(pass_events_df, all_frames_df, on='id',
                        how='left')  # # left join the frames onto the events. We now get several entries for each pass,
    filtered_lj_df = dc(lj_df[lj_df['location_y'].apply(lambda x: isinstance(x, list))].reset_index(
        drop=True))
    aggregated_frames_pass_events = make_threesixty(filtered_lj_df)
    pass_events_df = pd.merge(pass_events_df, aggregated_frames_pass_events, on='id', how='inner')    



    lj_df = pd.merge(all_events_df, all_frames_df, on='id',
                        how='left')  # # left join the frames onto the events. We now get several entries for each pass,
    filtered_lj_df = dc(lj_df[lj_df['location_y'].apply(lambda x: isinstance(x, list))].reset_index(
        drop=True))


    aggregated_frames_all_events = make_threesixty(filtered_lj_df, return_only_frame=True)
    all_events_df = pd.merge(all_events_df, aggregated_frames_all_events, on='id', how='inner')

    pass_events_df['numsecs'] = pass_events_df['minute']*60 + pass_events_df['second']
    all_events_df['numsecs'] = all_events_df['minute']*60 + all_events_df['second']

    pass_events_df = pass_events_df.sort_values(by=['numsecs'])
    all_events_df = all_events_df.sort_values(by=['numsecs'])

    #all_events_df = all_events_df[['id', 'numsecs', 'team', '360_frame_df', 'timestamp', 'type']]
    all_linked_event_candidates = all_events_df[all_events_df['type'].isin(['Ball Receipt*', 'Carry', 'Pressure', 'Ball Recovery', 'Dribble', 'Shot', 'Goal Keeper', 'Ball Recovery', 'Pass'])]
    
    linked_preceding_events_frames_flipped = []
    linked_preceding_events_ids = []
    prev_visible_area_points = []
    is_flipped = []
    sec_difference = []
    for (idx, pass_event) in pass_events_df.iterrows():
        if (len(pass_event['360_frame_df'])) < 6:
            linked_preceding_events_frames_flipped.append(None)
            linked_preceding_events_ids.append(None)
            is_flipped.append(None)
            sec_difference.append(None)
            prev_visible_area_points.append(None)
            continue
        pass_numsecs = pass_event.numsecs
        potential_linked_event_numsecs, last_preceding_event = None, None
        for (jdx, potential_linked_event) in all_linked_event_candidates.iterrows():
            potential_linked_event_numsecs = potential_linked_event.numsecs
            if potential_linked_event_numsecs < pass_numsecs:
                last_preceding_event = potential_linked_event
            else: break
        if last_preceding_event is None: 
            linked_preceding_events_frames_flipped.append(None)
            linked_preceding_events_ids.append(None)
            is_flipped.append(None)
            sec_difference.append(None)
            prev_visible_area_points.append(None)
            continue
        frame_of_rel_event = dc(last_preceding_event['360_frame_df'])

        pass_frame_centroid = calculate_centroid(pass_event['360_frame_df'].loc[~pass_event['360_frame_df'].is_keeper]['location'])
        rel_frame_centroid = calculate_centroid(frame_of_rel_event.loc[~frame_of_rel_event.is_keeper]['location'])
        rel_frame_centroid_flipped = calculate_centroid([[120-x, 80-y] for [x,y] in list(frame_of_rel_event.loc[~frame_of_rel_event.is_keeper]['location'])])

        try:
            if euclidean(pass_frame_centroid, rel_frame_centroid) > euclidean(pass_frame_centroid, rel_frame_centroid_flipped): should_flip = True
            else: should_flip = False
        except ValueError:
            linked_preceding_events_frames_flipped.append(None)
            linked_preceding_events_ids.append(None)
            is_flipped.append(None)
            sec_difference.append(None)
            prev_visible_area_points.append(None)
            continue

        #if pass_event['team'] == last_preceding_event['team']: should_flip = False
        #else: should_flip = True
        
        if should_flip:
                frame_of_rel_event['is_teammate'] = [not x for x in frame_of_rel_event['is_teammate']]
                frame_of_rel_event['location'] = [[120-x[0], 80-x[1]] for x in frame_of_rel_event['location']]
                frame_of_rel_event['location_x'] = [120-x for x in frame_of_rel_event['location_x']]
                frame_of_rel_event['location_y'] = [80-y for y in frame_of_rel_event['location_y']]
                last_preceding_event['points_in_visible_area'] = tuple([(120-x, 80-y) for (x, y) in last_preceding_event['points_in_visible_area']])
                is_flipped.append(True)
        else: is_flipped.append(False)
        pass_event['360_frame_df']['was_point_in_view_before'] = [[int(x), int(y)] in last_preceding_event['points_in_visible_area'] for [x,y] in pass_event['360_frame_df']['location']]
        td = pd.Timestamp(pass_event['timestamp']) - pd.Timestamp(last_preceding_event['timestamp'])
        sec_difference.append(td.seconds + td.microseconds/1e6)
        linked_preceding_events_frames_flipped.append(frame_of_rel_event)
        linked_preceding_events_ids.append(last_preceding_event['id'])
        prev_visible_area_points.append(last_preceding_event['points_in_visible_area'])


    pass_events_df['linked_preceding_event_360_frame'] = linked_preceding_events_frames_flipped
    pass_events_df['linked_preceding_event_id'] = linked_preceding_events_ids
    pass_events_df['is_flipped'] = is_flipped
    pass_events_df['sec_difference'] = sec_difference
    pass_events_df.dropna(subset=['linked_preceding_event_360_frame'], inplace=True)
    pass_events_df = pass_events_df.loc[pass_events_df.sec_difference < 2.5].reset_index(drop=True)
    pass_events_df = pass_events_df.apply(generate_trajectories, axis=1)

    pass_events_df = pass_events_df.dropna(subset=['360_frame_df'])
    
    sub_events_df = events['substitutions'] ### get df for substitution events

    formations = get_formations_for_all_minutes(match_id, sub_events_df, events, season_id, fifa_df, include_player_vectors=include_player_vectors)
    if formations is None: 
        raise ValueError('Unable to get formation for all minutes -- skip this match')
    possible_teams = list(formations['team'].unique())
    if formations is None: return None
    master_dataframe = pd.merge(pass_events_df, formations, on=['team', 'minute'], how='outer')
    formations['opposite_team'] = [flip_teams(team, possible_teams) for team in formations['team']]
    master_dataframe = pd.merge(master_dataframe, formations, left_on=['team', 'minute'], right_on=['opposite_team', 'minute'], suffixes=('', '_opp'), how='left')
    if include_player_vectors:
        assert [ff.player_vector.isna().any() for ff in master_dataframe.formatted_formation].count(True) == 0
    return master_dataframe.reset_index(drop=True).dropna(subset=['id', '360_frame_df'])


if __name__ == '__main__':
    get_pass_master_df(3788766, 43, include_player_vectors=True)