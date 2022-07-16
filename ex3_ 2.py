"""
Student Name: Dvora Cohen
Student ID: 015407190

==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.

import numpy as np

from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
print(X.shape)
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# Models definition
# -----------------
#
# We build a classification pipeline with a BernoulliRBM feature extractor and
# a :class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
# classifier.

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# Training
# --------
#
# The hyperparameters of the entire model (learning rate, hidden layer size,
# regularization) were optimized by grid search, but the search is not
# reproduced here because of runtime constraints.

from sklearn.base import clone

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 10

# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)

# %%
# Evaluation
# ----------

from sklearn import metrics

Y_pred = rbm_features_classifier.predict(X_test)
print(
    "Logistic regression using RBM features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
Y_pred = raw_pixel_classifier.predict(X_test)
print(
    "Logistic regression using raw pixel features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# The features extracted by the BernoulliRBM help improve the classification
# accuracy with respect to the logistic regression on raw pixels.

# %%
# Plotting
# --------

import matplotlib.pyplot as plt

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("100 components extracted by RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()



#------------------Import additional lib-----------------------------------------
from sklearn.metrics import precision_score
import time


# Making the array of the numbers
sqrs=[]
for _ in range(2,21):
    sqrs.append(_**2)

avg_p=[]
time_diff=[]
# --------------------------------------- q1----------------------------------------------
def q1():
    ap=float()
    def q1a(i):
        """
        Calculate the time of RBM-Logistic Pipeline method.
        :param i: the components.
        :return: time of the procedure.
        """
        # Training RBM-Logistic Pipeline
        t0 = time.time()
        rbm.n_components = i
        rbm_features_classifier.fit(X_train, Y_train)

        # Training Logistic regression

        logistic_classifier = linear_model.LogisticRegression(C=100.0)
        logistic_classifier.fit(X_train, Y_train)
        t1 = time.time()
        time_diff.append(t1 - t0)

        ap = precision_score(Y_test, logistic_classifier.predict(X_test), average='macro')
        avg_p.append(ap)
        ap = precision_score(Y_test, logistic_classifier.predict(X_test), average=None)
  
    num_components_array = np.array([])
    train_time_array = np.array([])
    rbm_average_precision_macro_array = np.array([])
    rbm_transform_shape_array = np.array([])
    rbm_intercept_shape_array = np.array([])
    x_train_shape_array = np.array([])
    x_test_shape_array = np.array([])
    for num in range(2, 21):
    
        rbm.n_components = num*num
        num_components_array = np.append(num_components_array, rbm.n_components)
        
        t0 = time.time()
        rbm_features_classifier.fit(X_train, Y_train)
        t1 = time.time()
        
        x_train_shape_array = np.append(x_train_shape_array, X_train.shape)
        x_test_shape_array = np.append(x_test_shape_array, X_test.shape)
        rbm_transform_shape_array = np.append(rbm_transform_shape_array, rbm.transform(X_train).shape)
        rbm_intercept_shape_array = np.append(rbm_intercept_shape_array, rbm.intercept_hidden_.shape)
        train_time_array = np.append(train_time_array, round(t1-t0, 2))

        # Training the Logistic regression classifier directly on the pixel
        
        # #############################################################################
        # Evaluation
       
        Y_pred = rbm_features_classifier.predict(X_test)

        rbm_average_precision_macro = metrics.precision_score(Y_test, Y_pred, average='macro')
        rbm_average_precision_macro_array = np.append(rbm_average_precision_macro_array, round(rbm_average_precision_macro, 2))    
       
        # #############################################################################
        # Plotting
        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(rbm.components_):
            plt.subplot(num, num, i + 1)
            plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('%i components extracted by RBM' %rbm.n_components, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        
        plt.show()
        
    def q2():
        plt.title("Precision avegarage as function of components number")
        plt.legend(["average precision"])   
        plt.xlabel("number of components")
        plt.ylabel("average precision")
        plt.hlines(ap,0,400,colors='pink',linewidth=2.0)
        plt.plot(num_components_array, rbm_average_precision_macro_array)
        plt.show()

        plt.title("Train time as function of components number")
        plt.legend(["train time"])   
        plt.xlabel("number of components")
        plt.ylabel("train time")
        plt.plot(num_components_array, train_time_array)
        plt.show()

    q2()
        
        
q1()

# --------------------------------------- q2 -------------------------------------------------------------
