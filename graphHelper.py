"""
This python file is used as a helper to
    - Get live testing results.
    - Create graphs for the "Comparative Evaluation and Discussion" part of the final report.
These results were then put as conclusions in our project report.
"""

from random import random
import pandas as pd
from matplotlib import pyplot as plt, spines
from matplotlib.patches import Patch
from pandas import DataFrame
import numpy as np

""" LIVE TESTING RESULTS PART"""
""" THIS PART SHOULD BE COMMENTED OUT WHEN THE GRAPH CREATION PART OF THE CODE IS BEGIN USED"""
TP = 22  # True Positive Count
FP = 0  # False Positive Count
TN = 50  # True Negative Count
FN = 19  # False Negative Count

ACC = (TP + TN) / (TP + FP + TN + FN)  # Calculate accuracy
ERR = (FP + FN) / (TP + FP + TN + FN)  # Calculate error rate
PRE = TP / (TP + FP)  # Calculate precision
REC = TP / (TP + FN)  # Calculate recall
SPE = TN / (TN + FP)  # Calculate specificity
F1S = (2 * TP) / (2 * TP + FP + FN)  # Calculate f1-Score
FNR = FN / (FN + TP)  # Calculate false negative rate
FPR = FP / (FP + TN)  # Calculate false positive rate

# Print Accuracy
print("Accuracy: ", end="")
print(ACC)

# Print error rate
print("Error Rate: ", end="")
print(ERR)

# Print precision
print("Precision: ", end="")
print(PRE)

# Print recall
print("Recall: ", end="")
print(REC)

# Print specificity
print("Specificity: ", end="")
print(SPE)

# Print f1-Score
print("f1-Score: ", end="")
print(F1S)

# Print false negative rate
print("False Negative Rate: ", end="")
print(FNR)

# Print false positive rate
print("False Positive Rate: ", end="")
print(FPR)


""" COMMENT THIS PART OUT TO CREATE GRAPHS """
"""color = ('#4d8963', '#69a583', '#e1b478', '#e0cc96', '#ec799a', '#9f0251',)  # Colors for graph bars
objects = ('XGBoost', 'AdaBoost', 'RF', 'LR', 'NB', 'k-NN')  # Names for each bar
y_pos = np.arange(len(objects))  # Create array of model names
performance = [2.55, 2.92, 2.75, 8.01, 10.73, 7.27]  # Metric values for each model
width = 0.25  # The width of the bars
plt.bar(y_pos, performance, align='center', color=color)  # Plot values with each model
plt.xticks(y_pos, objects)  # Align x values
plt.ylim(0, 12)  # This adds a little space at the top of the plot to compensate for the annotation
plt.ylabel('False Positive Rate (%)', fontsize=16)  # Create label
plt.gca().spines[['top', 'right']].set_visible(False)  # Get rid of top and rid lines of the graph

# Map the names to the colors
cmap = dict(zip(performance, color))  # Create dictionary

# Create the rectangles for the legend
patches = [Patch(color=v, label=k) for k, v in cmap.items()]

# Create the legend for the plot
plt.legend(title='Classification Models', labels=objects, handles=patches, bbox_to_anchor=(0.5, 1.15),
           loc='upper center', fancybox=True, shadow=False, ncol=6, edgecolor="white")

# Add the plot annotations
for y, x in zip(performance, y_pos):
    plt.annotate(f'{y}%\n', xy=(x, y), ha='center', va='center')
plt.show()  # Show the plot"""

""" INSERT ONE OF THE BELOW EVALUATION METRIC ARRAYS TO THE 'PERFORMANCE' VARIABLE TO CREATE THE GRAPH FOR IT """
# "ACC": [0.9946, 0.9934, 0.9936, 0.9762, 0.9575, 0.9824], [99.46, 99.34, 99.36, 97.62, 95.75, 98.24], Accuracy values of our model
# "ERR": [0.0053, 0.0065, 0.0063, 0.0237, 0.0424, 0.0175], [0.53, 0.65, 0.63, 2.37, 4.24, 1.75], Error rate values of our model
# "PRE": [0.9959, 0.9954, 0.9957, 0.9878, 0.9855, 0.9885], [99.59, 99.54, 99.57, 98.78, 98.55, 98.85], Precision values of our model
# "REC": [0.9978, 0.9969, 0.9969, 0.9847, 0.9660, 0.9911], [99.78, 99.69, 99.69, 98.47, 96.60, 99.11], Recall values of our model
# "SPE": [0.9744, 0.9707, 0.9725, 0.9198, 0.8926, 0.9272], [97.44, 97.07, 97.25, 91.98, 89.26, 92.72], Specificity values of our model
# "F1-S": [0.9969, 0.9962, 0.9963, 0.9862, 0.9757, 0.9898], [99.69, 99.62, 99.63, 98.62, 97.57, 98.98], f1-Score values of our model
# "FNR": [0.0021, 0.0030, 0.0030, 0.0152, 0.0339, 0.0088], [0.21, 0.30, 0.30, 1.52, 3.39, 0.88], False negative rate values of our model
# "FPR": [0.0255, 0.0292, 0.0275, 0.0801, 0.1073, 0.0727], [2.55, 2.92, 2.75, 8.01, 10.73, 7.27], False positive rate values of our model

