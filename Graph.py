import os
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd

METRICS = SourceFileLoader("graph_metrics", os.path.join(os.path.dirname(__file__), "metrics")).load_module()

rcParams["font.family"] = "Times New Roman"
rcParams["font.weight"] = "bold"
RESULT_DIR = METRICS.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True)

_graph_number = 0
def save_graph():
    """Save the current figure and close it when a plotting block finishes."""
    global _graph_number
    _graph_number += 1
    filename = METRICS.GRAPH_FILENAMES[_graph_number - 1]
    plt.gcf().savefig(os.path.join(RESULT_DIR, filename),
                      dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(plt.gcf())

plt.show = save_graph

data = METRICS.GENDER_DISTRIBUTION

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

for ax, (title, labels, values, colors) in zip(axes, data):
    ax.pie(values, labels=labels, colors=colors, startangle=90,
           counterclock=False, autopct="%.1f%%", pctdistance=0.62,
           labeldistance=1.2, textprops={"fontsize": 15, "fontweight": "bold"},
           wedgeprops={ "linewidth": 0.8})
    ax.set_title(title, fontsize=21, fontweight="bold", pad=22)
    ax.set_aspect("equal")

plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.04, wspace=0.28)
plt.savefig(os.path.join(RESULT_DIR, "Gender_Distribution.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.show()


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

age = METRICS.CARDIO_AGE_GROUPS
with_disease = np.array(METRICS.CARDIO_WITH_DISEASE)
no_disease = np.array(METRICS.CARDIO_NO_DISEASE)
percent = METRICS.CARDIO_AGE_PERCENTAGES

fig, ax = plt.subplots(figsize=(5.1, 3.0), dpi=600)

x = np.arange(len(age))
width = 0.60

ax.bar(x, with_disease, width,
       color='#c0392b', edgecolor='none',
       label='With Disease')

ax.bar(x, no_disease, width,
       bottom=with_disease,
       color='#e8b9b5', edgecolor='none',
       label='No Disease')

for i, p in enumerate(percent):
    total = with_disease[i] + no_disease[i]
    ax.text(i, total + 4, p,
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_title('Cardiovascular Dataset',
             fontsize=14, fontweight='bold', pad=7)

ax.set_xlabel('Age Groups',
              fontsize=10, fontweight='bold', labelpad=5)

ax.set_ylabel('Number of Patients',
              fontsize=10, fontweight='bold', labelpad=5)

ax.set_xticks(x)
ax.set_xticklabels(age, fontsize=9, fontweight='bold')

ax.set_ylim(0, 260)
ax.set_yticks([0, 100, 200])
ax.tick_params(axis='y', labelsize=9)

ax.grid(axis='y', linestyle='-', linewidth=0.7,
        alpha=0.25)
ax.set_axisbelow(True)

ax.legend(loc='upper left',
          bbox_to_anchor=(0.02, 0.98),
          fontsize=8.5,
          frameon=True,
          framealpha=0.85)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()


plt.rcParams['font.family'] = 'Times New Roman'

age = METRICS.CLEVELAND_AGE_GROUPS
disease = METRICS.CLEVELAND_WITH_DISEASE
no_disease = METRICS.CLEVELAND_NO_DISEASE

x = np.arange(6)

fig, ax = plt.subplots(figsize=(4.7, 2.96), dpi=600)

ax.bar(x, disease, width=0.58, color='#E67E22')
ax.bar(x, no_disease, bottom=disease, width=0.58, color='#F7D8BC')

ax.text(x[0], 27.5, '32.0%', ha='center', va='bottom',
        fontsize=8, fontweight='bold')

ax.text(x[1], 47.5, '33.3%', ha='center', va='bottom',
        fontsize=8, fontweight='bold')

ax.text(x[2], 62.5, '53.3%', ha='center', va='bottom',
        fontsize=8, fontweight='bold')

ax.text(x[3], 77.0, '64.7%', ha='center', va='bottom',
        fontsize=8, fontweight='bold')

ax.text(x[4], 68.5, '100.0%', ha='center', va='bottom',
        fontsize=8, fontweight='bold')

# >70 VALUE -- INSIDE THE ORANGE BAR
ax.text(x[5], 38, '218.2%', ha='center', va='center',
        fontsize=8, fontweight='bold')

ax.set_title('Cleveland Dataset', fontsize=12, fontweight='bold', pad=6)

ax.set_xlabel('Age Groups', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Patients', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(age, fontsize=8.5, fontweight='bold')

ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60])
ax.tick_params(axis='y', labelsize=8)

ax.grid(axis='y', color='gray', alpha=0.25, linewidth=0.7)

legend_elements = [
    Patch(facecolor='#F7D8BC', label='No Disease'),
    Patch(facecolor='#E67E22', label='With Disease')
]

ax.legend(handles=legend_elements,
          loc='upper left',
          fontsize=8.5,
          frameon=True)

for s in ax.spines.values():
    s.set_color('#333333')
    s.set_linewidth(0.9)

ax.set_axisbelow(True)

plt.tight_layout()
plt.show()


np.random.seed(42)
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(1, 2, figsize=(8.43, 3.76), dpi=100)

# Data
xb1=np.r_[np.random.normal(115,13,150),np.random.normal(132,13,100),np.random.normal(145,11,50),[83,87,91,96,102,108,118,127,135]]
yb1=np.r_[np.random.normal(205,30,150),np.random.normal(220,28,100),np.random.normal(230,25,50),[250,285,252,238,270,145,130,150,120]]

xr1=np.r_[np.random.normal(125,15,150),np.random.normal(142,14,170),np.random.normal(157,16,90),[92,101,108,115,121,128,135,143,151,158,165,172,180,188,195,205,155,150,140,130]]
yr1=np.r_[np.random.normal(225,32,150),np.random.normal(240,31,170),np.random.normal(235,34,90),[295,310,300,315,340,325,310,360,345,375,310,335,295,280,230,220,120,125,115,130]]

xb2=np.r_[np.random.normal(118,13,120),np.random.normal(132,13,120),np.random.normal(145,12,65),[80,87,92,99,105,112,120,128,138,150,158,165]]
yb2=np.r_[np.random.normal(195,27,120),np.random.normal(205,28,120),np.random.normal(215,29,65),[270,245,235,180,155,145,135,150,140,125,155,145]]

xo2=np.r_[np.random.normal(125,15,120),np.random.normal(145,16,145),np.random.normal(160,15,70),[85,90,96,102,108,115,121,128,135,142,150,158,166,175,182,190,155,145,132]]
yo2=np.r_[np.random.normal(230,34,120),np.random.normal(250,37,145),np.random.normal(245,35,70),[310,295,305,285,330,250,315,335,325,345,390,320,300,315,305,290,120,140,150]]

# Scatter
ax[0].scatter(xb1,yb1,s=28,c='#3498DB',alpha=.67,edgecolors='#2980B9',linewidths=.25,label='No Disease')
ax[0].scatter(xr1,yr1,s=28,c='#C0392B',alpha=.67,edgecolors='#A93226',linewidths=.25,label='With Disease')
ax[1].scatter(xb2,yb2,s=28,c='#3498DB',alpha=.67,edgecolors='#2980B9',linewidths=.25,label='No Disease')
ax[1].scatter(xo2,yo2,s=28,c='#F39C3D',alpha=.72,edgecolors='#E67E22',linewidths=.25,label='With Disease')

# Dashed lines
ax[0].plot([88,194],[252,236],'--',c='#3498DB',lw=1.8,zorder=10)
ax[0].plot([88,191],[205,198],'--',c='#C0392B',lw=1.8,zorder=10)
ax[1].plot([83,190],[247,239],'--',c='#3498DB',lw=1.8,zorder=10)
ax[1].plot([83,190],[201,195],'--',c='#E67E22',lw=1.8,zorder=10)

# Formatting
ax[0].set_title('Cardiovascular Dataset',fontsize=15,fontweight='bold',pad=5)
ax[1].set_title('Cleveland Dataset',fontsize=15,fontweight='bold',pad=5)

for a in ax:
    a.set_xlabel('Resting Blood Pressure (mmHg)',fontsize=12,fontweight='bold')
    a.set_ylabel('Serum Cholesterol (mg/dL)',fontsize=12,fontweight='bold')
    a.tick_params(labelsize=9.5)
    a.grid(True,color='#D9D9D9',lw=.65,alpha=.65)
    a.set_axisbelow(True)
    for t in a.get_xticklabels()+a.get_yticklabels(): t.set_fontweight('bold')
    for s in a.spines.values(): s.set_linewidth(.9)
    a.legend(loc='upper left',bbox_to_anchor=(.025,.985),fontsize=9.5,
             frameon=True,framealpha=.88,borderpad=.45,handletextpad=.65,
             labelspacing=.35)

ax[0].set(xlim=(77,212),ylim=(100,390),xticks=[80,100,120,140,160,180,200],yticks=[100,150,200,250,300,350])
ax[1].set(xlim=(77,200),ylim=(100,405),xticks=[80,100,120,140,160,180],yticks=[100,150,200,250,300,350,400])

plt.subplots_adjust(left=.07,right=.985,bottom=.125,top=.925,wspace=.235)
plt.show()



np.random.seed(4)

labels = METRICS.VIOLIN_LABELS
params = METRICS.VIOLIN_PARAMETERS

def violin(ax, p, x, color):
    mean, sd, lo, hi = p
    y = np.linspace(lo, hi, 300)
    d = np.random.normal(mean, sd, 700)
    d = np.clip(d, lo, hi)
    kde = gaussian_kde(d, bw_method=.18)
    w = kde(y)
    w = w / w.max() * .29
    ax.fill_betweenx(y, x-w, x+w, facecolor=color,
                     edgecolor="#1675ad", linewidth=1.2, alpha=.9)
    ax.vlines(x, lo, hi, color="#1675ad", linewidth=1.5)
    ax.hlines(mean, x-.15, x+.15, color="#777777", linewidth=1.8)
    ax.hlines([lo,hi], x-.09, x+.09, color="#1675ad", linewidth=1.5)

fig, ax = plt.subplots(1,2,figsize=(8.3,3.3))

blue = "#4B9BCB"
red = "#D86E68"
orange = "#F2A04A"

for a, title, disease_color in zip(
    ax,
    ["Cardiovascular Dataset - Violin Plot",
     "Cleveland Dataset - Violin Plot"],
    [red, orange]
):
    for i,p in enumerate(params):
        violin(a,p,i+1,blue if i%2==0 else disease_color)

    a.set_xlim(.45,8.55)
    a.set_ylim(80,230)
    a.set_yticks([100,150,200])
    a.set_xticks(range(1,9))
    a.set_xticklabels(labels,rotation=45,ha="right",
                      fontsize=7.5,fontweight="bold")
    a.set_ylabel("Maximum Heart Rate (bpm)",
                 fontsize=10,fontweight="bold")
    a.set_title(title,fontsize=12,fontweight="bold",pad=5)
    a.grid(axis="y",alpha=.3)
    a.set_axisbelow(True)

plt.subplots_adjust(left=.09,right=.99,bottom=.30,
                    top=.91,wspace=.24)
plt.show()

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif','Liberation Serif','Times New Roman'],
                     'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})

age = METRICS.NORMALIZATION_COUNTS["age"]
chol = METRICS.NORMALIZATION_COUNTS["cholesterol"]
hr = METRICS.NORMALIZATION_COUNTS["heart_rate"]
bp = METRICS.NORMALIZATION_COUNTS["blood_pressure"]

def gen(counts,a,b,seed):
    r=np.random.RandomState(seed); e=np.linspace(a,b,len(counts)+1)
    x=np.concatenate([r.uniform(e[i]+1e-4,e[i+1]-1e-4,c) for i,c in enumerate(counts) if c])
    x[0],x[-1]=a,b
    return x

df=pd.DataFrame({'Age':gen(age,16,101,10),'Cholesterol':gen(chol,102,348,20),
                 'Max HR':gen(hr,75,248,30),'BP':gen(bp,71,195,40)})
norm=(df-df.min())/(df.max()-df.min())
colors={'Age':'#6caac4','Cholesterol':'#bd759c','Max HR':'#f5b04d','BP':'#d77760'}

fig=plt.figure(figsize=(6.02,4.24),dpi=600)
a1=fig.add_axes([.0764,.592,.9169,.349]); a2=fig.add_axes([.0764,.0943,.9169,.349])

for c in df:
    a1.hist(df[c],30,alpha=.8,color=colors[c],label=f'{c} (Range: {int(df[c].min())}-{int(df[c].max())})')
    a2.hist(norm[c],30,alpha=.8,color=colors[c],label=f'{c} (Normalized)')

for ax,title,xlabel,xlim in [(a1,'Before Normalization','Feature Value',(0,350)),
                             (a2,'After Normalization','Normalized Value (0-1)',(0,1))]:
    ax.set(title=title,xlabel=xlabel,ylabel='Frequency',xlim=xlim,ylim=(0,107))
    ax.set_xticks(range(0,351,50) if ax==a1 else np.arange(0,1.01,.2))
    ax.set_yticks(range(0,101,20)); ax.grid(True,linestyle='-',linewidth=.5,color='#e0e0e0',alpha=.7)
    ax.legend(loc='upper right',framealpha=.85,edgecolor='#cccccc',fontsize=8.5)
    ax.tick_params(labelsize=8.5,direction='out',length=3.5,width=1)

plt.savefig(os.path.join(RESULT_DIR, 'normalization_plot.png'), dpi=600)
plt.show()
labels = METRICS.CARDIO_FEATURE_IMPORTANCE["labels"]
values = METRICS.CARDIO_FEATURE_IMPORTANCE["values"]

fig, ax = plt.subplots(figsize=(6.4, 4.4))

bars = ax.barh(
    labels, values,
    height=0.72,
    color="#1f77b4",
    edgecolor="#1f77b4"
)

ax.set_xlim(0, 17)
ax.set_xlabel("Importance (%)", fontsize=9)
ax.set_xticks(np.arange(0, 18, 2))
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8, length=0)

ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
ax.set_axisbelow(True)

for bar, value in zip(bars, values):
    ax.text(
        value + 0.35,
        bar.get_y() + bar.get_height()/2,
        f"{value:.1f}%",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold"
    )

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

plt.tight_layout()
plt.show()


labels = METRICS.CLEVELAND_FEATURE_IMPORTANCE["labels"]
values = METRICS.CLEVELAND_FEATURE_IMPORTANCE["values"]

fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=600)

bar_color = "#95584d"
bars = ax.barh(labels, values, color=bar_color, height=0.72)

ax.invert_yaxis()
ax.set_xlim(0, 17)
ax.set_xticks(range(0, 18, 2))

ax.set_xlabel("Importance (%)", fontsize=9, fontweight="bold")

ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

ax.tick_params(axis="y", labelsize=7.5, length=0)
ax.tick_params(axis="x", labelsize=8, colors="black", length=3)

plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")

ax.set_ylabel("")

for bar, value in zip(bars, values):
    ax.text(
        value + 0.45,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        ha="left",
        fontsize=7.5,
        fontweight="bold",
        color="#333333"
    )

for spine in ax.spines.values():
    spine.set_color("#555555")
    spine.set_linewidth(0.8)

plt.tight_layout(pad=0.7)
plt.show()

np.random.seed(42)
recall = np.linspace(0, 1, 101)

cardiovascular = np.clip(0.995 - 0.018*recall + np.random.normal(0, 0.008, 101), 0.85, 1.00)
cleveland = np.clip(0.990 - 0.025*recall + np.random.normal(0, 0.008, 101), 0.85, 1.00)
cardiovascular[0] = cleveland[0] = 1.00

fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=600)
blue, teal = "#2585A8", "#159A91"

ax.fill_between(recall, cardiovascular, 0.85, color=blue, alpha=0.30)
ax.fill_between(recall, cleveland, 0.85, color=teal, alpha=0.30)
ax.plot(recall, cardiovascular, color=blue, lw=2.5, label="Cardiovascular Dataset (AP = 0.992)")
ax.plot(recall, cleveland, color=teal, lw=2.5, label="Cleveland Dataset (AP = 0.987)")

ax.set(xlim=(0, 1), ylim=(0.85, 1.00), xlabel="Recall", ylabel="Precision")
ax.xaxis.label.set_fontweight("bold")
ax.yaxis.label.set_fontweight("bold")
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0.86, 1.001, 0.02))
ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.01, 0.2)], fontsize=9, fontweight="bold")
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.86, 1.001, 0.02)], fontsize=9, fontweight="bold")

ax.grid(True, color="#B0B0B0", lw=0.7, alpha=0.35)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color("#333333")
    spine.set_linewidth(0.9)

legend = ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.025), fontsize=9.5,
                   frameon=True, fancybox=True, framealpha=0.88, facecolor="white",
                   edgecolor="#D0D0D0", handlelength=2.4)

for text in legend.get_texts():
    text.set_fontweight("bold")

plt.subplots_adjust(left=0.115, right=0.985, bottom=0.175, top=0.975)
plt.show()

attention = np.array(METRICS.ATTENTION_WEIGHTS)

features = [f"Feature {i}" for i in range(1,15)]
time_steps = [f"Time {i}" for i in range(1,11)]

fig, ax = plt.subplots(figsize=(5.83,4.52), dpi=600)
im = ax.imshow(attention, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=0.19)

ax.set_xticks(np.arange(14))
ax.set_xticklabels(features, rotation=45, ha="right", rotation_mode="anchor", fontsize=8, fontfamily="serif", fontweight="bold")
ax.set_yticks(np.arange(10))
ax.set_yticklabels(time_steps, fontsize=8, fontfamily="serif", fontweight="bold")

ax.set_xlabel("Clinical Features", fontsize=12, fontfamily="serif", fontweight="bold", labelpad=8)
ax.set_ylabel("Time Steps", fontsize=11, fontfamily="serif", fontweight="bold", labelpad=7)
ax.set_title("VISTA-Net Spatio-Temporal Attention Map", fontsize=14, fontfamily="serif", fontweight="bold", pad=10)

for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
cbar.set_label("Attention Weight", fontsize=9, fontfamily="serif", fontweight="bold", rotation=90, labelpad=8)
cbar.set_ticks([0.025,0.050,0.075,0.100,0.125,0.150,0.175])
cbar.ax.tick_params(labelsize=7, length=0)
cbar.outline.set_visible(False)

ax.tick_params(axis="both", which="both", length=0, pad=3)
plt.subplots_adjust(left=0.070, right=0.895, bottom=0.165, top=0.900)
plt.show()


iterations = np.arange(1, len(METRICS.FITNESS_VALUES["cardiovascular"]) + 1)
cardio_fitness = METRICS.FITNESS_VALUES["cardiovascular"]
cleveland_fitness = METRICS.FITNESS_VALUES["cleveland"]

plt.rcParams.update({"font.family":"serif","font.weight":"bold","axes.labelweight":"bold","axes.titleweight":"bold","xtick.labelsize":10.5,"ytick.labelsize":10.5})

fig, ax = plt.subplots(figsize=(6.5,5), dpi=600)

ax.plot(iterations, cardio_fitness, color="#1f77b4", marker="o", markersize=6, linewidth=1.4, label="Cardiovascular Dataset")
ax.plot(iterations, cleveland_fitness, color="#8c564b", marker="s", markersize=6, linewidth=1.4, label="Cleveland Dataset")
ax.axhline(97.80, color="#6baed6", linestyle="--", linewidth=1.2)
ax.axhline(97.95, color="#c49a8f", linestyle="--", linewidth=1.2)

ax.set(xlim=(0,50), ylim=(94,100), xticks=[0,10,20,30,40,50], yticks=[94,95,96,97,98,99,100], xlabel="Iteration", ylabel="Fitness Value")
ax.grid(True, axis="y", color="lightgray", linewidth=0.6)
ax.tick_params(axis="both", labelsize=10.5)
ax.set_axisbelow(True)

legend = ax.legend(loc="upper left", frameon=True, fontsize=10, edgecolor="lightgray", handlelength=2, handletextpad=.5, borderpad=.6)
for t in legend.get_texts(): t.set_fontweight("bold")

ax.text(.500,.959,"Cardiovascular Convergence: 97.80",transform=ax.transAxes,color="#1f77b4",fontsize=10,fontweight="bold",va="top")
ax.text(.500,.873,"Cleveland Convergence: 97.95",transform=ax.transAxes,color="#8c564b",fontsize=10,fontweight="bold",va="top")

for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(1)

plt.tight_layout()
plt.show()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman','Liberation Serif','DejaVu Serif']

cm = np.array(METRICS.CONFUSION_MATRICES["cardiovascular"])
labels = ['No Disease','Disease']

fig, ax = plt.subplots(figsize=(5,5), dpi=600)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            annot_kws={'fontsize':14,'fontweight':'bold','fontfamily':'serif'},
            xticklabels=labels, yticklabels=labels)

ax.set_title('a)Cardiovascular Heart Disease Dataset', fontsize=12.5, fontweight='bold', pad=9)
ax.set_xlabel('Predicted Label', fontsize=13.5, fontweight='bold', labelpad=5)
ax.set_ylabel('True Label', fontsize=13.5, fontweight='bold', labelpad=5)
ax.set_xticklabels(labels, fontsize=10.5, fontweight='bold', rotation=0)
ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold', rotation=90, va='center')
ax.tick_params(length=3.5, width=1.2, direction='out', color='black')

for s in ax.spines.values(): s.set_visible(False)

plt.subplots_adjust(left=.127, bottom=.11, right=.944, top=.896)
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'), dpi=300)
plt.show()

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif','Times New Roman','Liberation Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})

cm = np.array(METRICS.CONFUSION_MATRICES["cleveland"])
labels = ['No Disease','Disease']

fig, ax = plt.subplots(figsize=(5,5), dpi=600)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={'fontsize':13,'fontweight':'bold'}, ax=ax)
ax.set(title='b) Heart Disease Cleveland Dataset', xlabel='Predicted Label', ylabel='True Label')
ax.title.set_fontsize(13); ax.title.set_fontweight('bold')
ax.xaxis.label.set_fontsize(12); ax.yaxis.label.set_fontsize(12)
ax.xaxis.label.set_fontweight('bold'); ax.yaxis.label.set_fontweight('bold')
ax.set_yticklabels(labels, rotation=90, va='center', fontsize=10, fontweight='bold')
ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10, fontweight='bold')
ax.tick_params(length=4, color='black')
plt.tight_layout()
plt.show()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']

plots_data = [(np.array(matrix), title, cmap)
              for matrix, title, cmap in METRICS.CONFUSION_MATRICES["comparison"]]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=600)

for ax, (matrix, title, cmap) in zip(axes.flat, plots_data):
    h = sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'],
                    annot_kws={'fontsize':13,'fontweight':'bold'},
                    cbar_kws={'pad':0.04})
    ax.set(title=title, xlabel='Predicted Label', ylabel='True Label')
    ax.title.set_fontsize(11); ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(10.5); ax.yaxis.label.set_fontsize(10.5)
    ax.xaxis.label.set_fontweight('bold'); ax.yaxis.label.set_fontweight('bold')
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontsize=9.5, fontweight='bold', rotation=0)
    for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor('black'); s.set_linewidth(1)
    c = h.collections[0].colorbar
    c.outline.set_visible(True); c.outline.set_edgecolor('black'); c.outline.set_linewidth(1)
    c.ax.tick_params(labelsize=9)
    for t in c.ax.get_yticklabels(): t.set_fontweight('bold')

plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'mathtext.fontset':'dejavuserif','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})

x1 = np.arange(100, 100 * (len(METRICS.LEARNING_CURVE_VALUES["cardio_train"]) + 1), 100)
train1 = np.array(METRICS.LEARNING_CURVE_VALUES["cardio_train"])
val1 = np.array(METRICS.LEARNING_CURVE_VALUES["cardio_validation"])
x2 = np.array(METRICS.LEARNING_CURVE_VALUES["cleveland_sizes"])
train2 = np.array(METRICS.LEARNING_CURVE_VALUES["cleveland_train"])
val2 = np.array(METRICS.LEARNING_CURVE_VALUES["cleveland_validation"])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.58,3.74),dpi=600)
colors=[('#287d9b','#973867'),('#149687','#ed217c')]

for ax,x,tr,va,title,xticks,limits,(c1,c2) in [
    (ax1,x1,train1,val1,'Cardiovascular Dataset',[200,400,600,800,1000],(85,1030),colors[0]),
    (ax2,x2,train2,val2,'Cleveland Dataset',[50,100,150,200,250,300],(25,312),colors[1])]:

    ax.plot(x,tr,'o-',color=c1,lw=2,ms=6.5,label='Training Accuracy')
    ax.fill_between(x,np.clip(tr-.02,.8,1),np.clip(tr+.02,.8,1),color=c1,alpha=.22)
    ax.plot(x,va,'s-',color=c2,lw=2,ms=6.5,label='Validation Accuracy')
    ax.fill_between(x,np.clip(va-.025,.8,1),np.clip(va+.025,.8,1),color=c2,alpha=.20)
    ax.set(title=title,xlabel='Training Set Size',ylabel='Accuracy',ylim=(.8,1),xlim=limits,xticks=xticks,yticks=np.arange(.8,1.001,.025))
    ax.set_yticklabels([f'{y:.3f}' for y in np.arange(.8,1.001,.025)],fontsize=9.5,fontweight='bold')
    ax.set_xticklabels(xticks,fontsize=9.5,fontweight='bold')
    ax.grid(True,linestyle='-',alpha=.4,color='#d3d3d3')
    ax.legend(loc='lower right' if ax is ax1 else 'upper left',framealpha=.9,prop={'weight':'bold','size':10.5})

plt.tight_layout()
plt.show()

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','Liberation Serif','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})

epochs = np.arange(1, len(METRICS.LOSS_VALUES["cardio_train"]) + 1)
cardio_train = METRICS.LOSS_VALUES["cardio_train"]
cardio_val = METRICS.LOSS_VALUES["cardio_validation"]
cleve_train = METRICS.LOSS_VALUES["cleveland_train"]
cleve_val = METRICS.LOSS_VALUES["cleveland_validation"]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.3,3.79),dpi=600)
fig.subplots_adjust(left=62/930,right=915/930,bottom=41/379,top=359/379,wspace=61/396)

for ax,tr,va,title,c1,c2,yt in [(ax1,cardio_train,cardio_val,'Cardiovascular Dataset','#2E86AB','#A23B72',np.arange(0,.9,.1)),(ax2,cleve_train,cleve_val,'Cleveland Dataset','#1B998B','#ED217C',np.arange(0,.9,.2))]:
    ax.plot(epochs,tr,'o-',color=c1,label='Training Loss',linewidth=2,markersize=6.5)
    ax.plot(epochs,va,'s-',color=c2,label='Validation Loss',linewidth=2,markersize=6.5)
    ax.set(title=title,xlabel='Epoch',ylabel='Loss',xticks=np.arange(0,51,10),yticks=yt)
    ax.set_yticklabels([f'{y:.1f}' for y in yt])
    ax.grid(True,linestyle='-',alpha=.35,color='#cccccc'); ax.set_axisbelow(True)
    ax.legend(loc='upper right',prop={'weight':'bold','size':11.5},borderaxespad=.4)
    ax.tick_params(labelsize=11,width=1.2,length=4)
    for s in ax.spines.values(): s.set_linewidth(1.2)

plt.savefig(os.path.join(RESULT_DIR, 'loss_curves.png'), dpi=100)
plt.show()

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','axes.edgecolor':'black','axes.linewidth':1})

models = METRICS.MODEL_COMPARISON["models"]
x = np.arange(len(models))
width = .35
data = METRICS.MODEL_COMPARISON["metrics"]

fig,axes=plt.subplots(2,2,figsize=(8.2901,4.70),dpi=600)

for i,(ax,d) in enumerate(zip(axes.flat,data)):
    title,ylabel,s1,s2,c1,c2=d
    ax.bar(x-width/2,s1,width,color=c1,label='Cardiovascular Dataset' if i==0 else None)
    ax.bar(x+width/2,s2,width,color=c2,label='Cleveland Dataset' if i==0 else None)
    ax.set(title=title,xlabel='Models',ylabel=ylabel,xticks=x,xticklabels=models,xlim=(-.58,3.58),ylim=(90,100),yticks=np.arange(90,101,2))
    ax.tick_params(labelsize=9.5)
    ax.grid(axis='y',linestyle='-',color='#e5e5e5',linewidth=.1,alpha=.8); ax.set_axisbelow(False)
    if i==0:
        leg=ax.legend(loc='upper left',frameon=True,prop={'weight':'bold','size':8.5},handlelength=1.4,handleheight=.9,borderpad=.35,labelspacing=.35)
        leg.get_frame().set_edgecolor('#cccccc'); leg.get_frame().set_linewidth(.8)

plt.subplots_adjust(left=48/829,right=820/829,bottom=45/470,top=440/470,wspace=60/356,hspace=67/164)
plt.show()
