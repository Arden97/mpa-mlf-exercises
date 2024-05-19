# Import usefull libraries
import pandas as pd # import pandas for working with data sets
import matplotlib.pyplot as plt # import pyplot for plotting the graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Function to plot the graph
def plot_the_graph(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Data preprocessing
pd_dataset = pd.read_csv('/path/to/data.zip')
# turn our .csv dataset in to 2 dimensional data structure
df = pd.DataFrame(pd_dataset)
# drop useless columns
df_new = df.drop(columns=['Unnamed: 0', 'time [s]'])

range_n_clusters = range(2, 11)

# # Elbow method
wcss = []
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i) # trying different numbers of clusters  
    kmeans.fit(df_new) # fit our data in to a model
    wcss.append(kmeans.inertia_) # the sum of squared distances of samples to their nearest cluster center
# plotting the results onto a line graph to observe the 'elbow'
plot_the_graph(x=range_n_clusters, y=wcss, xlabel='Number of clusters', ylabel='WCSS', title='Elbow Method')

# Silhouette method
silhouette_avg_scores = []
for i in range_n_clusters:
    clusterer = KMeans(n_clusters=i) # trying different numbers of clusters 
    cluster_labels = clusterer.fit_predict(df_new) # fit our data into model and predict, where each sample belongs to
    silhouette_avg = silhouette_score(df_new, cluster_labels) # returns the measure of how similar samples are to its own cluster compared to other clusters
    silhouette_avg_scores.append(silhouette_avg) # append the average silhouette score in to a list
# plotting the results onto a line graph to observe the silhouette scores
plot_the_graph(x=range_n_clusters, y=silhouette_avg_scores, title='Silhouette Method', xlabel='Number of clusters', ylabel='Average Silhouette Score')

plt.show()

