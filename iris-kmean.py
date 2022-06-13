from numpy import array, int32
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.utils import shuffle

# import some data to play with
iris = datasets.load_iris()
X = iris.data 
y = iris.target
names = iris.feature_names
X, y = shuffle(X, y, random_state=42)

model = KMeans(n_clusters=3, random_state=42) 
iris_kmeans = model.fit(X)

