import torch

tragets = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_station': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9
}

a = [1, 0, 0, 0]
b = [1, 0, 0, 0]
print(a == b)
