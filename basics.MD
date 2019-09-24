* Generalization of models means how well a model performs on new previously unseen data.
For models, the generalization error aka test error to be as small as possible.
* The generalization error is defined as the expected value of the error on a new input.
* Statistical Learning Theory provides a framework why why ML models perform well on test data when models have been
trained on the training data.
* Assumption that is made is i.i.d = > the distributions generating data are assumed to be identically distributed but
test and training data are independent. Identically distributed means drawn from same distribution.

Problems with ML models can be underfitting and overfitting.
* Underfitting -> model not able to achieve low error on training data. ut
* Overfitting -> too large gap on test and training data.
One way we can control models overfitting or underfitting problem is with models capacity.
*  capacity of model - is ability to fit multiple variety of functions for modelling.
* Low capacity -> underfiting occurs
* High capacity -> Overfitting occurs because the model remembers properties of training data.
* One way to control the capacity of a learning algorithm is by choosing its hypothesis space, the set of functions that
 the learning algorithm is allowed to select as being the solution
 *Occam Razor -> This principle states that among competing hypotheses that explain known observations equally well, one
should choose the “simplest” one.
* Statistical Learning Theory provides various means of quantifying model capacity- > Vapnik-Chervonenkis (VC) Dimension is one
well known way of measuring capacity of binary classifiers.
* The most important results in statistical learning theory show that the discrepancy between training error and
generalization error is bounded from above by a quantity that grows as the model capacity grows but
shrinks as the number of training examples increases
* The bound that SLT provides a justification that ML algorithms can work.
* The ideal model is an oracle that simply knows the true probability distribution  that generates the data.
* The error incurred by an oracle making predictions from the true distribution p (x , y ) is called the Bayes error.
* No Free Lunch Theorem -> Any ML algorithm not better than any other. averaged over all possible data generating
 distributions, every classification algorithm has the same error rate when classifying previously unobserved points.
* Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its
training error.
* Hyperparameters - setting that we can use to control the behaviour of learning algorithms
* validation set set taken from taining data that is not used for traning but for validating the model before applying to
the testing test.
* Typically, one uses about 80% of the training data for training and 20% for validation.
* the k-fold cross-validation - a partition of the dataset is formed by splitting it into k non-overlapping subsets.
* bias(parameter) = ExpectedValue(paramter) - actualValueofParamter
* For bernouli and Gausssian distributions, the mean is the unbiased estimator
* Variance -> measures how much an estimator varies as function of the data sample
* Bias and variance measure two different sources of error in an estimator.
* Bias measures the expected deviation from the true value of the function or parameter.
* Variance provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause.
* What happens when we are given a choice between two estimators, one with more bias and one with more variance?  Trade off between bias and variance is handled by cross-validation method.
* other way of handling the trade-off is using MSE b/n bias and variance
* As model capacity increases, bias decreases and variance increases.
* Consistency means as the number of the sample size increases the point estimate of a parameter convergs to the actual parameter value.
* Consistency ensures that the bias induced by the estimator diminishes as the number of data examples grows.
* maximum likelihood as an attempt to make the model distribution match the empirical distribution p̂ data .
* Maximum Likelihood Estimate : principle from which we can derive specific functions that are good estimators for different models.
* for Linear regression : MLE derives the MSE, ie, maximizing the log likelihood wrt w === minimizing the MSE wrt w
* maximum likelihood is often considered the preferred estimator to use for machine learning. When the number
of examples is small enough to yield overfitting behavior, regularization strategies
such as weight decay may be used to obtain a biased version of maximum likelihood
that has less variance when training data is limited.
* Bayesian vs frequentist statistics =>
    * frequentist statistics and approaches based on estimating a single value of θ, then making all predictions thereafter based on that one
estimate.
    * Bayesian statistics : Another approach  that considers all possible values of θ when making a prediction.
* Bayesian estimation offers two important differences. First, unlike the maximum likelihood approach that makes
predictions using a point estimate of θ, the Bayesian approach is to make predictions
using a full distribution over θ. the second difference from the maximum likelihood approach is due to the contribution of the Bayesian prior distribution.
* Bayesian methods typically generalize much better when limited training data is available, but typically suffer from high computational cost when the number of
training examples is large.
* Linear Regression to Classification problem by using the logistic sigmoid function to squash the output of the linear function into the
interval (0, 1) and interpret that value as a probability
* SVM - t(W)X+b > 0 -> class 1 and t(W)X+b<0 -> class 0
* Gaussian Kernel aka RBF ->  performs kind of template matching  -> SVMS are enhanced by the kernel trick
* A major drawback to kernel machines is that the cost of evaluating the decision function is linear in the number of training examples, because the i-th example
contributes a term α i k(x , x ( i ) ) to the decision function.
* Support vector machines are able to mitigate this by learning an α vector that contains mostly zeros. Training examples with non zero learning α are support vectors
* Kernel machines also suffer from a high computational cost of training when the dataset is large. Kernel machines struggle to generalize with the mostly used kernels and
can not generalize thus deep learning methods came to resolve this issues with the SV.
* Unsupervised learning algorithms - > PCA, KMeans
* One difficulty pertaining to clustering is that the clustering problem is inherently ill-posed, in the sense that
 there is no single criterion that measures how well a clustering of the data corresponds to the real world
* Stochastic Gradient Descent
* The core idea in deep learning is that we assume that the data was generated by the composition of factors or features, potentially at multiple levels in a hierarchy.
Reasons Why DL is introduced

* Curse of Dimensionality -> ML problems become difficult when number of dimensions is high
* Local Constancy -> the function we learn should not change very much within a small region and this fails for large scale tasks. but DL introduces additional
priors to reduce generalization error.
* Manifold Learning
*
