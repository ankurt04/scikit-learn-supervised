# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:09:57 2017

@author: Ankurt.04
"""

"""
Fashion MNIST Data


Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 
Each pixel has a single pixel-value associated with it, indicating the lightness or darkness 
of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 
0 and 255. The training and test data sets have 785 columns. The first column consists of 
the class labels (see above), and represents the article of clothing. The rest of the columns 
contain the pixel-values of the associated image.

•To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, 
where i and j are integers between 0 and 27. The pixel is located on row i and column j of 
a 28 x 28 matrix.
•For example, pixel31 indicates the pixel that is in the fourth column from the left, 
and the second row from the top, as in the ascii-diagram below. 

Labels
Each training and test example is assigned to one of the following labels:
• 0 T-shirt/top
• 1 Trouser
• 2 Pullover
• 3 Dress
• 4 Coat
• 5 Sandal
• 6 Shirt
• 7 Sneaker
• 8 Bag
• 9 Ankle boot 
TL;DR: - Each row is a separate image 
- Column 1 is the class label. - Remaining columns are pixel numbers (784 total). - 
Each value is the darkness of the pixel (1 to 255)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('fashion-mnist_train_20000.csv')
#print(train_df.head)


FEATURES = list(train_df)
del FEATURES[0] 
#print(FEATURES)

def build_data():
    """
    Build training and test sets here in the form of 2D numpy array
    """
    
    X = np.array(train_df[FEATURES].values)
    #print(X)
    y = np.array(train_df['label'].values.tolist())
    #print(y)
    X = preprocessing.scale(X)
    
    return X, y

def ml_fit_testing(X, y):
    """
    Select the algorithm and train the data. In ML, training means to fit. 
    """
    
    """Algo 1 =  K-neighbors"""
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    
    # Create a k-NN classifier with 7 neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    # Print the accuracy
    print(knn.score(X_test, y_test))
    
    return X_test, knn


def predict_test_data(X_pred, knn):
    """
    Run the alog on the test data and verify its performance
    """
    y_pred = knn.predict(X_pred)
    print("Category label of the product is: ", y_pred)
        
    
def visual_disp(X_pred):
    """
    Create plot of the output if required
    """
    #plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
    #plt.show()
    
    # Select and reshape a row
    im = X_pred
    im_sq = np.reshape(im, (28, 28))
    #print(im_sq)

    # Plot reshaped data (matplotlib.pyplot already loaded as plt)
    plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
    plt.show()



def main():
    X, y = build_data()
    X_test, knn = ml_fit_testing(X, y)
    
    #value to be predicted and graphed
    X_pred = X_test[2, :]
    #print(X_pred)
    
    predict_test_data(X_pred, knn)
    visual_disp(X_pred)
    
    
if __name__ == '__main__':
    main()