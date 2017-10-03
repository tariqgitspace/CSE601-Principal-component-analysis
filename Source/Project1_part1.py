import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as L
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import seaborn as sns

text_file = 'pca_a.txt'

#plt.style.use('ggplot')
#plt.style.use('dark_background')
#plt.style.use('seaborn-dark-palette')


file_values = []
with open(text_file,'r') as f:
	for line in f:
		values = line.split('\t')
		file_values.append(values)

numrows = len(file_values)
numcols = len(file_values[0])

X=np.array(file_values)

labels=X[:,numcols-1]
labelset=set(labels)

X=np.delete(X,numcols-1,1)
X=np.array(X).astype(np.float)


Y = np.mean(X, axis=0)
Z=np.subtract(X,Y)


covar=(1/(numrows-1))*np.dot(Z.T,Z)
eigvalues,eigvectors=L.eig(covar)


idx = np.argsort(eigvalues)[::-1]
principal_components=[]
principal_components.append(eigvectors[:,idx[0]]) ##highest eigen value
principal_components.append(eigvectors[:,idx[1]])  ## 2nd highest eigen value
principal_components=np.array(principal_components)

final=np.dot(Z,principal_components.T)

f, ax = plt.subplots(figsize=(10,5))
#ax.set_color_cycle(['blue','red', 'black', 'yellow'])
for name in labelset:
	x = final[labels[:]==name,0]  #all places where label is met and then plot (1st eigen*x) as x axis 
	y = final[labels[:]==name,1]  #all places where label is met and then plot (2nd eigen*x) as y axis
	ax.scatter(x, y,marker='o',label=name) ## marker='o'

plt.title("PCA plot for : "+str(text_file))
plt.legend(loc='upper left',ncol=1,fontsize=12)
plt.show()

plt.show()    

####################################   t-SNE   ##################################################3

tsne2 = TSNE(n_components=2,random_state=0)
tsne2_results = tsne2.fit_transform(X)
#print(tsne2_results)
first = tsne2_results[:,0] 
second = tsne2_results[:,1]

f, ax = plt.subplots(figsize=(10,5))
#ax.set_color_cycle(['blue','red', 'black', 'yellow'])
for name in labelset:
    x = first[labels[:]==name]  #all places where label is met and then plot (1st eigen*x) as x axis 
    y = second[labels[:]==name]  #all places where label is met and then plot (2nd eigen*x) as y axis
    ax.scatter(x, y,marker='o',label=name)
plt.title("t-SNE plot for : "+str(text_file))
plt.legend(loc='upper left',ncol=1,fontsize=12)
plt.show()


################################### SVD ################################################
u,s,v=np.linalg.svd(Z.T)
principal_components_svd=u[:,[0,1]]

result=X.dot(principal_components_svd)

f, ax = plt.subplots(figsize=(10,5))
#ax.set_color_cycle(['blue','red', 'black', 'yellow'])
for name in labelset:
	x = result[labels[:]==name,0]
	y = result[labels[:]==name,1]
	ax.scatter(x, y,label=name)
plt.title("SVD plot for : "+str(text_file))
plt.legend(loc='upper left',ncol=1,fontsize=12)
plt.show()