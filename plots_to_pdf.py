import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sb
from pathlib import Path

sb.set(style='whitegrid')
path = Path(os.path.abspath(__file__)).parent
fname = 'plots.pdf'
df = pd.read_csv(path.joinpath('.csv'))
dates = df['DATE']

def visual(data, dates, title_str):
    fig, ax = plt.subplots(figsize=(15,7))
    ax = ax.set_xticklabels(labels=dates, rotation=90, ha='right')
    plt.tight_layout()
    plot = sb.lineplot(data=data, palette='BuGn_r', linewidth=2.5).set_title(title_str)

with PdfPages(path.joinpath(fname)) as pdf_pages:
    for col in df.columns.values:
        plot = visual(df, dates, col)
        pdf_pages.savefig(plot)
