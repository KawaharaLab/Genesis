import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


BASE_PATH = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"

if __name__ == "__main__":


    plot_path_base = BASE_PATH + '/main/data/picked_up/plots'
    os.makedirs(plot_path_base, exist_ok=True)
    names = ["BABY_CAR","Dino_3","Dino_4","Dino_5"]

    

    path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up', 'csv')
    
    for name in names:#os.listdir(path):
        plot_path = os.path.join(plot_path_base,f'{name}')
        os.makedirs(plot_path, exist_ok=True)
        for target in ['hard', 'medium', 'soft']:
            deform_path =  os.path.join(path, name, f'Elastic/{target}/{name}_Elastic_deform_{target}.csv')
            force_path = os.path.join(path, name, f'Elastic/{target}/{name}_Elastic_{target}.csv')
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            with open(deform_path, 'r') as f:
                deform_csv = pd.read_csv(f, header=0)

            with open(force_path, 'r') as f:
                df = pd.read_csv(f, header=0)
                
    
            # Deformation plot
            axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='.', color='tab:blue', linewidth=0.5)
            axs[0].set_xlabel('Time Step')
            axs[0].set_ylabel('Deformation Metric')
            axs[0].set_ylim(0, 0.6)
            axs[0].set_title(f'Object: {name} | Target: {target}')
            axs[0].grid(True)

            # Force components plot
            force_columns = ['left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz']
            for col in force_columns:
                axs[1].plot(df['step'], df[col], marker='.', label=col)
            axs[1].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 2], marker='.', linestyle='-', color='black', label='grip_force', linewidth=0.5)
            axs[1].set_ylim(-30, 25)
            axs[1].set_xlabel('Time Step')
            axs[1].set_ylabel('Force (N)')
            axs[1].set_title('Force Components Over Time')
            axs[1].grid(True)
            axs[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f'{name}_{target}'), dpi=300, bbox_inches='tight')
            print(f"Saved plot -> {plot_path}")
            # plt.show()