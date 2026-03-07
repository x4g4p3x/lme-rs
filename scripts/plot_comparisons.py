import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Data extracted from COMPARISONS.md
data = []

# Sleepstudy (LMM)
# R, Rust, Python, Julia
sleepstudy = {
    'Dataset': 'Sleepstudy (LMM)',
    '(Intercept)': [251.405, 251.405, 251.405, 251.405],
    'Days': [10.467, 10.467, 10.467, 10.467]
}
for i, lang in enumerate(['R (lme4)', 'Rust (lme-rs)', 'Python (lme_python)', 'Julia (MixedModels)']):
    data.append({'Dataset': 'Sleepstudy', 'Language': lang, 'Parameter': '(Intercept)', 'Estimate': sleepstudy['(Intercept)'][i]})
    data.append({'Dataset': 'Sleepstudy', 'Language': lang, 'Parameter': 'Days', 'Estimate': sleepstudy['Days'][i]})

# Grouseticks (Poisson)
grouseticks = {
    'Dataset': 'Grouseticks (Poisson GLMM)',
    '(Intercept)': [-0.4132, -0.4132, -0.4132, -0.4134],
    'YEAR': [-0.0322, -0.0322, -0.0322, -0.0322],
    'HEIGHT': [-0.1344, -0.1344, -0.1344, -0.1344]
}
for i, lang in enumerate(['R (lme4)', 'Rust (lme-rs)', 'Python (lme_python)', 'Julia (MixedModels)']):
    for param in ['(Intercept)', 'YEAR', 'HEIGHT']:
        data.append({'Dataset': 'Grouseticks', 'Language': lang, 'Parameter': param, 'Estimate': grouseticks[param][i]})

# CBPP (Binomial)
cbpp = {
    '(Intercept)': [-1.3983, -1.3605, -1.3605, -1.3985],
    'period2': [-0.9919, -0.9761, -0.9761, -0.9923],
    'period3': [-1.1282, -1.1110, -1.1110, -1.1287],
    'period4': [-1.5797, -1.5596, -1.5596, -1.5803]
}
for i, lang in enumerate(['R (lme4)', 'Rust (lme-rs)', 'Python (lme_python)', 'Julia (MixedModels)']):
    for param in ['(Intercept)', 'period2', 'period3', 'period4']:
        data.append({'Dataset': 'CBPP', 'Language': lang, 'Parameter': param, 'Estimate': cbpp[param][i]})

# Penicillin (Crossed)
penicillin = {
    '(Intercept)': [22.9722, 22.9722, 22.9722, 22.9722]
}
for i, lang in enumerate(['R (lme4)', 'Rust (lme-rs)', 'Python (lme_python)', 'Julia (MixedModels)']):
    data.append({'Dataset': 'Penicillin', 'Language': lang, 'Parameter': '(Intercept)', 'Estimate': penicillin['(Intercept)'][i]})

# Pastes (Nested)
pastes = {
    '(Intercept)': [60.0533, 60.0533, 60.0533, 60.0533]
}
for i, lang in enumerate(['R (lme4)', 'Rust (lme-rs)', 'Python (lme_python)', 'Julia (MixedModels)']):
    data.append({'Dataset': 'Pastes', 'Language': lang, 'Parameter': '(Intercept)', 'Estimate': pastes['(Intercept)'][i]})

df = pd.DataFrame(data)

# Set style
sns.set_theme(style="whitegrid", context="talk")
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3'] # R, Rust, Python, Julia

fig, axes = plt.subplots(3, 2, figsize=(20, 20))
axes = axes.flatten()

datasets = ['Sleepstudy', 'Penicillin', 'Pastes', 'Grouseticks', 'CBPP']
titles = ['Sleepstudy (Linear Mixed Model)', 'Penicillin (Crossed Random Effects)', 'Pastes (Nested Random Effects)',
          'Grouseticks (Poisson GLMM)', 'CBPP (Binomial GLMM)']

for i, (dataset, title) in enumerate(zip(datasets, titles)):
    ax = axes[i]
    subset = df[df['Dataset'] == dataset]
    
    sns.barplot(
        data=subset, 
        x='Parameter', 
        y='Estimate', 
        hue='Language', 
        ax=ax,
        palette=colors,
        alpha=0.9
    )
    
    ax.set_title(title, pad=20, fontsize=18, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Fixed Effect Estimate')
    
    # We only need one legend
    if i == 0:
        ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()
        
    # Rotate x labels for CBPP and Grouseticks
    if dataset in ['CBPP', 'Grouseticks']:
        ax.tick_params(axis='x', rotation=45)
        
    # Add exact value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, rotation=90, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
fig.suptitle('Numerical Parity of Mixed-Effects Models Across ecosystem implementations', fontsize=24, fontweight='bold')

plt.savefig('examples/comparison_chart.png', dpi=300, bbox_inches='tight')
print("Saved chart to examples/comparison_chart.png!")
