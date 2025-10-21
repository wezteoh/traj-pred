import numpy as np

from src.utils.drawing import create_frames_from_trajectory, create_video_from_frames


def visualize_trajectory(trajectory: np.ndarray):
    """
    visualize a soccer trajectory
    """
    frames = create_frames_from_trajectory(trajectory, game="football")
    create_video_from_frames(frames, "/Users/wzteoh/Downloads/football/football.mp4", fps=5)
    return frames


if __name__ == "__main__":
    trajectory = np.load("/Users/wzteoh/Downloads/football/train_clean.p", allow_pickle=True)
    trajectory = trajectory[0]
    trajectory = np.concatenate([trajectory[:, 1:], trajectory[:, :1]], axis=1)
    visualize_trajectory(trajectory)
