
def compute_reward(car, map_image):
    """
    Compute the reward for the current car state.
    Basic reward shaping:
    - +1 for staying alive (per frame)
    - +10 for staying on the road (black pixels)
    - -100 for going off-road (non-black)
    """

    reward = 1.0

    try:
        pixel_color = map_image.get_at((int(car.x), int(car.y)))[:3]
    except IndexError:
        return -100.0

    if sum(pixel_color) < 60:
        reward += 10.0
    else:
        reward -= 100.0

    return reward