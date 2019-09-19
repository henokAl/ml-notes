* Generalization of models means how well a model performs on new previously unseen data.
For models, the generalization error aka test error to be as small as possible.
* The generalization error is defined as the expected value of the error on a new input.
* Statistical Learning Theory provides a framework why why ML models perform well on test data when models have been
trained on the training data.
* Assumption that is made is i.i.d = > the distributions generating data are assumed to be identically distributed but
test and training data are independent. Identically distributed means drawn from same distribution.

Problems with ML models can be underfitting and overfitting.
* Underfitting -> model not able to achieve low error on training data.
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