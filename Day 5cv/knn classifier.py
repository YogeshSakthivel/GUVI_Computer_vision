import numpy as np       
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits=datasets.load_digits()


X=digits.images
Y=digits.target

n_samples =len(X)
X=X.reshape((n_samples, -1))

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.5,shuffle=False)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)


Y_pred=knn.predict(X_test)

print("KNN Accuracy:",metrics.accuracy_score(Y_test,Y_pred))

imags_and_prediction=list(zip(digits.images[n_samples // 2:], Y_pred))
for index,(image,prediction) in enumerate (imags_and_prediction[:4]):
    plt.subplot(1,4, index +1)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title(f'pred:{prediction}')
plt.show()    