import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import io

dataset = """sepal.length	sepal.width	petal.length	petal.width	variety
5.1	3.5	1.4	.2	Setosa
4.9	3	1.4	.2	Setosa
4.7	3.2	1.3	.2	Setosa
4.6	3.1	1.5	.2	Setosa
5	3.6	1.4	.2	Setosa
5.4	3.9	1.7	.4	Setosa
4.6	3.4	1.4	.3	Setosa
5	3.4	1.5	.2	Setosa
4.4	2.9	1.4	.2	Setosa
4.9	3.1	1.5	.1	Setosa
5.4	3.7	1.5	.2	Setosa
4.8	3.4	1.6	.2	Setosa
4.8	3	1.4	.1	Setosa
4.3	3	1.1	.1	Setosa
5.8	4	1.2	.2	Setosa
5.7	4.4	1.5	.4	Setosa
5.4	3.9	1.3	.4	Setosa
5.1	3.5	1.4	.3	Setosa
5.7	3.8	1.7	.3	Setosa
5.1	3.8	1.5	.3	Setosa
5.4	3.4	1.7	.2	Setosa
5.1	3.7	1.5	.4	Setosa
4.6	3.6	1	.2	Setosa
5.1	3.3	1.7	.5	Setosa
4.8	3.4	1.9	.2	Setosa
5	3	1.6	.2	Setosa
5	3.4	1.6	.4	Setosa
5.2	3.5	1.5	.2	Setosa
5.2	3.4	1.4	.2	Setosa
4.7	3.2	1.6	.2	Setosa
4.8	3.1	1.6	.2	Setosa
5.4	3.4	1.5	.4	Setosa
5.2	4.1	1.5	.1	Setosa
5.5	4.2	1.4	.2	Setosa
4.9	3.1	1.5	.2	Setosa
5	3.2	1.2	.2	Setosa
5.5	3.5	1.3	.2	Setosa
4.9	3.6	1.4	.1	Setosa
4.4	3	1.3	.2	Setosa
5.1	3.4	1.5	.2	Setosa
5	3.5	1.3	.3	Setosa
4.5	2.3	1.3	.3	Setosa
4.4	3.2	1.3	.2	Setosa
5	3.5	1.6	.6	Setosa
5.1	3.8	1.9	.4	Setosa
4.8	3	1.4	.3	Setosa
5.1	3.8	1.6	.2	Setosa
4.6	3.2	1.4	.2	Setosa
5.3	3.7	1.5	.2	Setosa
5	3.3	1.4	.2	Setosa
7	3.2	4.7	1.4	Versicolor
6.4	3.2	4.5	1.5	Versicolor
6.9	3.1	4.9	1.5	Versicolor
5.5	2.3	4	1.3	Versicolor
6.5	2.8	4.6	1.5	Versicolor
5.7	2.8	4.5	1.3	Versicolor
6.3	3.3	4.7	1.6	Versicolor
4.9	2.4	3.3	1	Versicolor
6.6	2.9	4.6	1.3	Versicolor
5.2	2.7	3.9	1.4	Versicolor
5	2	3.5	1	Versicolor
5.9	3	4.2	1.5	Versicolor
6	2.2	4	1	Versicolor
6.1	2.9	4.7	1.4	Versicolor
5.6	2.9	3.6	1.3	Versicolor
6.7	3.1	4.4	1.4	Versicolor
5.6	3	4.5	1.5	Versicolor
5.8	2.7	4.1	1	Versicolor
6.2	2.2	4.5	1.5	Versicolor
5.6	2.5	3.9	1.1	Versicolor
5.9	3.2	4.8	1.8	Versicolor
6.1	2.8	4	1.3	Versicolor
6.3	2.5	4.9	1.5	Versicolor
6.1	2.8	4.7	1.2	Versicolor
6.4	2.9	4.3	1.3	Versicolor
6.6	3	4.4	1.4	Versicolor
6.8	2.8	4.8	1.4	Versicolor
6.7	3	5	1.7	Versicolor
6	2.9	4.5	1.5	Versicolor
5.7	2.6	3.5	1	Versicolor
5.5	2.4	3.8	1.1	Versicolor
5.5	2.4	3.7	1	Versicolor
5.8	2.7	3.9	1.2	Versicolor
6	2.7	5.1	1.6	Versicolor
5.4	3	4.5	1.5	Versicolor
6	3.4	4.5	1.6	Versicolor
6.7	3.1	4.7	1.5	Versicolor
6.3	2.3	4.4	1.3	Versicolor
5.6	3	4.1	1.3	Versicolor
5.5	2.5	4	1.3	Versicolor
5.5	2.6	4.4	1.2	Versicolor
6.1	3	4.6	1.4	Versicolor
5.8	2.6	4	1.2	Versicolor
5	2.3	3.3	1	Versicolor
5.6	2.7	4.2	1.3	Versicolor
5.7	3	4.2	1.2	Versicolor
5.7	2.9	4.2	1.3	Versicolor
6.2	2.9	4.3	1.3	Versicolor
5.1	2.5	3	1.1	Versicolor
5.7	2.8	4.1	1.3	Versicolor
6.3	3.3	6	2.5	Virginica
5.8	2.7	5.1	1.9	Virginica
7.1	3	5.9	2.1	Virginica
6.3	2.9	5.6	1.8	Virginica
6.5	3	5.8	2.2	Virginica
7.6	3	6.6	2.1	Virginica
4.9	2.5	4.5	1.7	Virginica
7.3	2.9	6.3	1.8	Virginica
6.7	2.5	5.8	1.8	Virginica
7.2	3.6	6.1	2.5	Virginica
6.5	3.2	5.1	2	Virginica
6.4	2.7	5.3	1.9	Virginica
6.8	3	5.5	2.1	Virginica
5.7	2.5	5	2	Virginica
5.8	2.8	5.1	2.4	Virginica
6.4	3.2	5.3	2.3	Virginica
6.5	3	5.5	1.8	Virginica
7.7	3.8	6.7	2.2	Virginica
7.7	2.6	6.9	2.3	Virginica
6	2.2	5	1.5	Virginica
6.9	3.2	5.7	2.3	Virginica
5.6	2.8	4.9	2	Virginica
7.7	2.8	6.7	2.0	Virginica
6.3	2.7	4.9	1.8	Virginica
6.7	3.3	5.7	2.1	Virginica
7.2	3.2	6	1.8	Virginica
6.2	2.8	4.8	1.8	Virginica
6.1	3	4.9	1.8	Virginica
6.4	2.8	5.6	2.1	Virginica
7.2	3	5.8	1.6	Virginica
7.4	2.8	6.1	1.9	Virginica
7.9	3.8	6.4	2.0	Virginica
6.4	2.8	5.6	2.2	Virginica
6.3	2.8	5.1	1.5	Virginica
6.1	2.6	5.6	1.4	Virginica
7.7	3	6.1	2.3	Virginica
6.3	3.4	5.6	2.4	Virginica
6.4	3.1	5.5	1.8	Virginica
6	3	4.8	1.8	Virginica
6.9	3.1	5.4	2.1	Virginica
6.7	3.1	5.6	2.4	Virginica
6.9	3.1	5.1	2.3	Virginica
5.8	2.7	5.1	1.9	Virginica
6.8	3.2	5.9	2.3	Virginica
6.7	3.3	5.7	2.5	Virginica
6.7	3	5.2	2.3	Virginica
6.3	2.5	5	1.9	Virginica
6.5	3	5.2	2.0	Virginica
6.2	3.4	5.4	2.3	Virginica
5.9	3	5.1	1.8	Virginica
"""

dataframe = pd.read_csv(io.StringIO(dataset), sep="\t")

X = dataframe[['sepal.length', 'sepal.width']].values
y = dataframe['variety'].values

classes, y_numericalValues = np.unique(y, return_inverse=True)

knearestneighbors = KNeighborsClassifier(n_neighbors=7)
knearestneighbors.fit(X, y_numericalValues)

x_minimum, x_maximum = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_minimum, y_maximum = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_minimum, x_maximum, 0.01),
                     np.arange(y_minimum, y_maximum, 0.01))

Z = knearestneighbors.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['magenta', 'lightblue', 'gold'])
cmap_bold  = ListedColormap(['purple', 'turquoise', 'yellow'])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_numericalValues, cmap=cmap_bold, edgecolor='k', s=50)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Decision Boundary for K-nearest neighbor with K = 7')

legend_patches = [mpatches.Patch(color=cmap_bold.colors[i], label=classes[i]) for i in range(len(classes))]
plt.legend(handles=legend_patches, title='Classes', loc='lower right')
plt.xticks(np.arange(4.5, 8.1, 0.5))
plt.yticks(np.arange(2.0, 4.6, 0.5))
plt.show()