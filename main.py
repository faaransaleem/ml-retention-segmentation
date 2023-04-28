# this will help in making the Python code more structured automatically (good coding practice)
# %load_ext nb_black

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)

# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import pdist, cdist

# to perform k-means clustering, compute metric
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

#!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to perform PCA
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# loading the dataset
from google.oauth2 import service_account

import time

start=time.time()
credentials = service_account.Credentials.from_service_account_file('creds.json')

All_Sales_data=pd.read_gbq('''
select * from `technical_models.vw_mp_o_customer_performance`
where mp_ae_total_app_sessions>2
'''
                           ,credentials=credentials
                          )


end=time.time()
print(end-start)

df=All_Sales_data.copy()
df.shape

# viewing a random sample of the dataset
df.sample(n=10, random_state=1)

df.info()

df.describe(include="all").T

# lets check total null values 
df.isnull().sum()

df=df.fillna(0)

# lets check duplicate observations
df.duplicated().sum()

subset_scaled_df= df.drop(['number','phone_original_customer','rn'],axis=1)
sc = StandardScaler()
subset_scaled_df = pd.DataFrame(
    sc.fit_transform(subset_scaled_df),
    columns=subset_scaled_df.columns,
)
subset_scaled_df.head()

clusters=range(1,20)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction=model.predict(subset_scaled_df)
    distortion=sum(np.min(cdist(subset_scaled_df, model.cluster_centers_, 'euclidean'), axis=1)) / subset_scaled_df.shape[0]
                           
    meanDistortions.append(distortion)

    print('Number of Clusters:', k, '\tAverage Distortion:', distortion)

plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Distortion')
plt.title('Selecting k with the Elbow Method', fontsize=20)

# checking silhoutte score

sil_score = []
cluster_list = list(range(2,10))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((subset_scaled_df))
    #centers = clusterer.cluster_centers_
    score = silhouette_score(subset_scaled_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    
visualizer = SilhouetteVisualizer(KMeans(4, random_state = 1))
visualizer.fit(subset_scaled_df)    
visualizer.show();

visualizer = SilhouetteVisualizer(KMeans(6, random_state = 1))
visualizer.fit(subset_scaled_df)    
visualizer.show();

kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(subset_scaled_df)

df['K_means_segments'] = kmeans.labels_
subset_scaled_df['K_means_segments'] = kmeans.labels_

import pickle
pickle.dump(kmeans,open("customer_segment_kmeans_2020_oct_31_v1.pkl","wb"))

cluster_profile = df.groupby('K_means_segments').mean()
cluster_profile['count_in_each_segments'] = df.groupby('K_means_segments').count().values

