# Machine Learning theory

Notebooks explaining the statistics behind a set of machine learning models, including code for building the models from scratch (without using packages)

## 1. Linear regression and overfitting
This notebook fits a set of polynomial regressions with varying degrees on periodic data. The generated plots provide a visualize the concepts of over- and under-fitting. Next, a regularized linear regression is implemented, a method that should prevent overfitting, and cross validation is used to determine the optimal hyperparameters (the regularization coefficient and the degree of the polynomial). The final model obtained shows the usefullness of regularized linear regression in this example, as the selected hyperparameters result in a well fitting prediction function.

## 2. Bayesian linear regression
This notebook focuses on Bayesian linear regression and consists of two parts. 

### 2.1 Sequantial updating
In the first part of the notebook, the basics of Bayesian learning are visualized by applying it for a simple linear model and displaying the likelihood, prior, posterior and dataspace for several sequential draws. It shows that the posterior becomes more and more centered around the true values as more data is observed, resulting in a data space that resembles the true relation. A zero-mean isotropic conjugate Gaussian prior is used 

### 2.2 Predictive distribution
The second part of the notebook a Bayesian linear regression model is implemented to fit periodic data, also using a zero-mean isotropic Gaussian prior. Instead of visualizing the results of several sequential draws, one update is applied and the Bayesian predicitve distribution is visualized. It shows the predictive mean and standard deviation, to provide insight on how this depends on the observed data.

## 3. Neural Network
In this notebook, a one-hidden-layer nerual network with weight decay is implemented from scratch and applied on binary classification problem. Instead of using a package, the model is created in several steps.
* The parameters are initialized
* The predicted probabilities for a set of input data is calculated
* The gradient of the loss function is calculated
  * This takes the input, the predicted probabilities, the actual target values and the parameters as input
  * It returns the gradients of the loss function wrt. parameters
* The parameters are updated with Stochastic Gradient Decent (SGD) that includes weight decay
* This is repeated for a certain number of epochs to train the model. 

These steps are used on the training data, once with and once without weight decay, and the trained models are consequently used to predict for a test set. The results are visualized and the accuracy is calculate (0.84 and 0.82 respectively).

## 4. Binary classification
This notebook a basic version of a set of Machine Learning algorithms on predicting wheather a client will default on its credit card or not. First, three algorithms are implemented in their most basic form, with pre-determined hyperparameters values. Their performance is evaluated through the accuracy score and the confusion matrix. The models used are:
* logistic regression
* AdaBoost
* random forest

After implementing the basic models, the random forest is expanded. GridSearch is used to find the optimal parameters, which is done for a model on which feature engineering is applied as well as on one where there isn't.

## 5. Multiclass classification
In this notebook, two classfication algorithms are applied on classifying the handwritten digits from the MNIST dataset. 

### 5.1 Multiclass logistic regression
First, multiclass logistic regression is implemented. This is done without the use of packages, but through first determining the gradient equations of the models and consequently writing a function that implements stochastic gradient descent. 

### 5.2 Multilayer perceptron
Similarly, the multilayer perceptron model is implemented from scratch, using forward propogation, backward propogation and SGD (for a set of learning rates).

## 6. Gaussian Process regression and Support Vector Machines
In this notebook, two kernel-based algorithms are implemented from scratch.
* Guassian process regression
* Support vector machine, applied on a binary classification problem


