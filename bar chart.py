import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Baseline, Assistive', 'LexiS, Assistive', 'SynS, Assistive', 'SemS, Assistive']
sari_keep = [57.71, 55.86, 53.55, 53.91]
sari_add = [5.31, 4.37, 4.30, 4.59]
sari_delete = [48.67, 47.70, 48.25, 49.55]

# Set position of bar on Y axis
r = np.arange(len(labels))

# Create horizontal bars
plt.barh(r, sari_keep, color='b', edgecolor='grey', label='KEEP')
plt.barh(r, sari_add, color='orange', left=sari_keep, edgecolor='grey', label='ADD')
plt.barh(r, sari_delete, color='g', left=np.array(sari_keep) + np.array(sari_add), edgecolor='grey', label='DELETE')

# Add labels
plt.xlabel('Scores')

plt.yticks(r, labels)
plt.title('Three shots')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
plt.show()