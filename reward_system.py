def compute_reward(car, map_image):
    """
    Compute the reward for the current car state.
    Reward shaping:
    - +1 for staying alive (per frame)
    - +10 for being on road (black pixels)
    - +speed * 0.1 to encourage movement
    - -100 for going off-road (non-black)
    """

    reward = 1.0

    try:
        pixel_color = map_image.get_at((int(car.x), int(car.y)))[:3]
    except IndexError:
        car.alive = False
        return -100.0

    if sum(pixel_color) < 60:
        reward += 10.0
        reward += car.speed * 0.1 
    else:
        reward -= 100.0
        car.alive = False

    return reward
