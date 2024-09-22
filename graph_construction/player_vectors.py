from unidecode import unidecode
import numpy as np
import pandas as pd
from copy import deepcopy as dc
import re

def get_player_vector(team, name, fifa_df=None) -> np.array:
    """
    Fetch the player vector from the associated fifa_df for the player 'name' of nationality 'team'. If no player can be identified, 0 is returned.
    """
    if team == 'South Korea': team = 'Korea Republic'

    ### First, retrieve all players of the nationality, specified in the team name above.
    subset = fifa_df[fifa_df['nationality_name'] == team]

    found_player = False
    
    ### check if we can find only one player with this last name
    selected_player = subset.loc[subset.short_name.map(unidecode).map(lambda x: x.lower()).str.contains(f'{unidecode(name.split(" ")[-1]).lower()}', flags=re.IGNORECASE)]
    num_found_players = len(selected_player)
    if num_found_players == 1: found_player = True

    ### check if we have a match for the full name
    if not found_player:
        selected_player = subset.loc[
            subset.long_name.map(unidecode).map(lambda x: x.lower()).str.contains(f'{unidecode(name).lower()}',
                                                                                  flags=re.IGNORECASE)]
        num_found_players = len(selected_player)
        if num_found_players == 1: found_player = True

    ### check if we have a match for first and last part of long name
    if not found_player:
        selected_player = subset.loc[(subset.long_name.map(unidecode).map(lambda x: x.lower()).str.contains(
            f'{unidecode(name.split(" ")[-1]).lower()}', flags=re.IGNORECASE)) & (
                                         subset.long_name.map(unidecode).map(lambda x: x.lower()).str.contains(
                                             f'{unidecode(name.split(" ")[0]).lower()}', flags=re.IGNORECASE))]
        num_found_players = len(selected_player)
        if num_found_players == 1: found_player = True

    ### check if any part of the name is unique
    if not found_player:
        for name_part in name.split(" "):
            selected_player = subset.loc[subset.short_name.map(unidecode).map(lambda x: x.lower()).str.contains(
                f'{unidecode(name_part).lower()}', flags=re.IGNORECASE)]
            num_found_players = len(selected_player)
            if num_found_players == 1:
                found_player = True
                break

    ### if not, go over all parts of the name and find an interesection of the found parts.
        ### First, try going over combinations of two parts of the name.
    split_name = name.replace('-', ' ').split(" ")
    doubles = [f'{unidecode(name[i-1]).lower()} {unidecode(name[i]).lower()}' for i in range(1, len(split_name))]
    if not found_player:
        found = []
        for double in doubles:
            selected_player = dc(subset.loc[subset.long_name.map(unidecode).map(lambda x: x.lower()).str.contains(
                f'{unidecode(double).lower()}', flags=re.IGNORECASE)])
            num_found_players = len(selected_player)
            if num_found_players == 1:
                found_player = True
                break
    if not found_player:
        selection_list = []
        for name_part in name.replace('-', ' ').split(" "):
            selected_player = dc(subset.loc[subset.long_name.map(unidecode).map(lambda x: x.lower()).str.contains(
                f'{unidecode(name_part).lower()}', flags=re.IGNORECASE)])
            num_found_players = len(selected_player)
            if num_found_players == 1:
                found_player = True
                break

    if not found_player:
        return 0
        ### failsafe

    arr = selected_player.drop(['long_name', 'short_name', 'nationality_name'], axis=1)
    arr = np.array(arr).reshape(1, -1)
    return arr
