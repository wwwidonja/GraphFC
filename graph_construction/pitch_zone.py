def return_pitch_zone_index(point):
    """
    return index of pitch control zone according to positional play map.
    """
    x, y = point[0], point[1]

    if 0<=x<18 and 0<=y<=22: return 0 ## top left space
    elif 0<=x<=18 and 22<=y<=58: return 1 #left penalty box
    elif 0<=x<=18 and 58<=y<=80: return 2 ## bottom left space
    elif 18<=x<=39 and 0<=y<=22: return 3 ##top mid defensive
    elif 39<=x<=60 and 0<=y<=22: return 4 ## top right defensive
    elif 18<=x<=60 and 22<=y<=30: return 5 ## top half-space defensive
    elif 18<=x<=60 and 30<=y<=50: return 6 ## middle defensive
    elif 18<=x<=60 and 50<=y<=58: return 7 ## bottom half-space defensive
    elif 18<=x<=39 and 58<=y<=80: return 8 ## bottom mid defensive
    elif 39<=x<=60 and 58<=y<=80: return 9 ##bottom right defensive
    elif 60<=x<=81 and 0<=y<=22: return 10 ##top left offensive
    elif 81<=x<=102 and 0<=y<=22: return 11 ##top mid offensive
    elif 60<=x<=102 and 22<=y<=30: return 12 ##top half-space offensive
    elif 60<=x<=102 and 30<=y<=50: return 13 ##middle offesnive
    elif 60<=x<=102 and 50<=y<=58: return 14 ## bottom half-space offensive
    elif 60<=x<=81 and 58<=y<=80: return 15 ## bottom left offensive
    elif 81<=x<=102 and 58<=y<=80: return 16 ## bottom mid offensive
    elif 102<=x<=120 and 0 <= y <= 22: return 17 ## top right space
    elif 102<=x<=120 and 22<=y<=58: return 18 ##right penalty box
    elif 102<=x<=120 and 58<=y<=80: return 19 ##bottom left space
    return None