import pickle
import os

def save_path_if_high_reward(path_history, total_reward, save_dir="saved_paths"):
    """
    Saves the path history if the total reward exceeds key thresholds.
    - > 3000: Save to path_3000.pkl
    - > 4000: Save to path_4000.pkl
    - > 5000: Save to path_5000.pkl
    """

    os.makedirs(save_dir, exist_ok=True)

    if total_reward > 5000:
        filename = os.path.join(save_dir, "path_5000.pkl")
    elif total_reward > 4000:
        filename = os.path.join(save_dir, "path_4000.pkl")
    elif total_reward > 3000:
        filename = os.path.join(save_dir, "path_3000.pkl")
    else:
        return

    with open(filename, "wb") as f:
        pickle.dump(path_history, f)
    print(f"✅ Path saved to {filename} (Reward: {total_reward:.2f})")
