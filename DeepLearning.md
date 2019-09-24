Deep Forward Network aka MultiLayer Perceptron FeedForward NN.
* Aim: f(x) -> y find a function that transform sample X to label y
* No feedback connection. If feedback is added RNN is created
* the convolutional networks used for object recognition from photos are a specialized kind of feedforward network.
* can be considered as DAG f3(f2(f1)) f1 is Layer 1, f2 is Layer 2 ....
* The layer of this DAG or chain is the depth, hence that is why they are called Deep Learning
* training data does not show the desired output for each of these layers, these layers are called hidden layers.
* Each hidden layer of the network is typically vector-valued. The dimensionality of these hidden layers determines the width of the model
* feedforward networks as function approximation machines that are designed to achieve statistical generalization,
occasionally drawing some insights from what we know about the brain, rather than as models of brain function.

How FeedForward Network resolve limitations of Linear and Non Linear models?
* Linear models ,for example logistic regression and linear regression, are appealing because they may fit efficiently
and reliably, either in closed form or with convex optimization. Linear models also have the obvious defect that the model
capacity is limited to linear functions, so the model cannot understand the interaction between any two input variables.

* To make linear models available to non linear functions of the input x , we transform the input x to higher space  φ(x), and apply linear models to this
* How to chose φ(x) 
    * use generic RBF, usually based on local smoothness
    * manually engineer φ(x) based on application area image/language/speech
    * DL strategy is to actually learn φ(x)
* Feedforward networks have introduced the concept of a hidden layer, and this requires us to choose the activation functions that will
be used to compute the hidden layer values.
* XOR problem shows how linear model can not handle the XOR function, this is solved by FeedForward NN

Gradient Based Learning
* Difference betweeen linear models and neural networks is that the nonlinearity of a neural network causes most
 interesting loss functions to become non-convex.
 * Using MSE -Mean Sqaured Error or MAE -Mean Abolute Error- for Gradient based optimization is not good choice. Cross entropy cost functions are better with 
 Gradient based learning optimizations 
 * z = tr(W)*h + b -> unnormalized log probabilities
 * softMax function is found by expo and normalizing z exp(z)/sum(exp(z))
Gradient Decent Optimization 
    * w(t+1) =w(t) - learning_rate*chnage_in_error_wrt_w(t)
    * w ( t+1) = w ( t ) − η∇E(w ( t ) )
    * ∇E(w ( τ ) ) is evaluated for the training data
    * Batch methods -> use all the data for the update step 
    * At each step the weight vector is moved in the direction of the greatest rate of decrease of the error function,
and so this approach is known as gradient descent or steepest descent. It's poor method compared to Conjugate QN
    * Faster alternatives to simple GD optimization are conjugate gradient and Quasi-Newton
    * Unlike gradient descent, these algorithms have the property that the error function always decreases at 
    each iteration unless the weight vector has arrived at a local or global minimum.
    
Online GD [Stochastic GD , Sequential GD]
* Makes update to the weight vector based on one data input
* w ( τ +1) = w ( τ ) − η∇E n (w ( τ ) ).
* Stochastic GD chooses random input for training the model ie. samples are selected randomly 

Why online method better than batch GD? 
* online methods handle redundancy in the data much more efficiently.
* on-line gradient descent escape the possibility of being stuck in local minima

Error Propagation (Backprop)
* efficient technique for evaluating the gradient of an error function E(w) for a feed-forward neural network.
* this can be achieved using a local message passing scheme in which information is sent alternately forwards and backwards through the network
* Efficiency : a[j]= sum(w [j][i] *z[i] ) O(W) steps
