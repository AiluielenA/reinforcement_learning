import pandas as pd

class GameLogger:
    def __init__(self):
        self.logs = []

    def log_step(self, step, rewards, energy_levels, packages_delivered):
        self.logs.append({
            "step": step,
            "rewards": rewards,
            "robot_1_energy": energy_levels[0],
            "robot_2_energy": energy_levels[1],
            "packages_delivered": packages_delivered
        })

    def save_logs(self, filename="game_logs.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Logs saved to {filename}")
