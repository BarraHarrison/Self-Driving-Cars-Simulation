def compute_reward(car, map_image):
    reward = 1.0

    try:
        pixel_color = map_image.get_at((int(car.x), int(car.y)))[:3]
    except IndexError:
        car.alive = False
        return -100.0

    if sum(pixel_color) < 60:
        reward += 10.0
        reward += car.speed * 0.1

        if len(car.path_history) > 2:
            recent = car.path_history[-2:]
            delta_x = abs(recent[1][0] - recent[0][0])
            delta_y = abs(recent[1][1] - recent[0][1])
            if delta_x < 2 and delta_y < 2:
                reward -= 2.0
    else:
        reward -= 100.0
        car.alive = False

    return reward
