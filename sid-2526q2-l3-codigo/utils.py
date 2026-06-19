import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def draw_history(history, title, save_path=None):
    if isinstance(history, dict):
        data = pd.DataFrame(history)
    else:
        data = pd.DataFrame({title: history})
    data.index.name = 'Epoch'

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data)

    plt.title(title + ' Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    # guardamos imagen
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
