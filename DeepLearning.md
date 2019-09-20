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
* Linear models for example logistic regression and linear regression, are appealing because they may be fit efficiently
and reliably, either in closed form or with convex optimization. Linear models also have the obvious defect that the model
capacity is limited to linear functions, so the model cannot understand the interaction between any two input variables.

* To make linear models available to non linear functions of the input x , we transform the input x to higher space  φ(x), and apply linear models to this
* How to chose φ(x) -> use generic RBF, usually based on local smoothness
                    -> manually engineer φ(x) based on application area image/language/speech
                    -> DL strategy is to actually learn φ(x)
* Feedforward networks have introduced the concept of a hidden layer, and this requires us to choose the activation functions that will
be used to compute the hidden layer values.
* XOR problem shows how linear model can not handle the XOR function, this is solved by FeedForward NN
*