import numpy as np       
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits=datasets.load_digits()

X=digits.images
Y=digits.target

n_samples =len(X)
X=X.reshape((n_samples, -1))

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.5,shuffle=False)

clf =svm.SVC(gamma=0.001)
clf.fit(X_train,Y_train)

Y_pred= clf.predict(X_test)
print("Classification report:\n",metrics.classification_report(Y_test,Y_pred))

imags_and_prediction=list(zip(digits.images[n_samples // 2:], Y_pred))
for index,(image,prediction) in enumerate (imags_and_prediction[:8]):
    plt.subplot(1,8, index +1)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title(f'pred:{prediction}')
plt.show()    