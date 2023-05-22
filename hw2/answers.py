r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. $\frac{\partial Y}{\partial X}$:<br>
    A. for each entry $i,j$ in $Y$ and entry $k,m$ in $X$ there is an entry $\frac{\partial Y_{i,j}}{\partial X_{k,m}}$.<br>
    so the shape is $(N,D_{out},N,D_{in})$.<br>
    B. Yes, this Jacobian is sparse. Y's i'th row is only affected by X's i'th row and not it's other rows. (each row in X is an input and the corresponding row in Y is the output). So every entry $(i,j,k,m)$ in the tensor for which $i \neq k$ is zero.<br>
    C. No. I will show how to calculate $\frac{\partial L}{\partial X}$.<br>
    $row_i(\delta X) = row_i(\delta Y) @ \frac{\partial (row_i(\delta Y))}{\partial (row_i(\delta X))}$, shapes: $(1,D_{in}) = (1,D_{out}) @ (D_{out},D_{in})$
    So for $M$ of shape $(N,D_{out},D_{in})$ defined such that $M[i,:,:]$ is $\frac{\partial (row_i(\delta Y))}{\partial (row_i(\delta X))}$ we get 
    $\delta X = \delta Y @ M$

1. $\frac{\partial Y}{\partial W}$:<br>
    A. for each entry $i,j$ in $Y$ and entry $k,m$ in $W$ there is an entry $\frac{\partial Y_{i,j}}{\partial W_{k,m}}$.<br>
    so the shape is $(N,D_{out},D_{out},D_{in})$.<br>
    B. Yes, this Jacobian is sparse. Y's i'th column is only affected by W's i'th row and none of it's other rows. So every entry $(i,j,k,m)$ in the tensor for which $j \neq k$ is zero.<br>
    C. No. I will show how to calculate $\frac{\partial L}{\partial W}$.<br>
    $row_i(\delta W) = col_i(\delta Y) @ \frac{\partial (col_i(\delta Y))}{\partial (row_i(\delta W))}$, shapes: $(1,D_{in}) = (1,N) @ (N,D_{in})$
    So for $M$ of shape $(D_{out},N,D_{in})$ defined such that $M[i,:,:]$ is $\frac{\partial (col_i(\delta Y))}{\partial (row_i(\delta W))}$ we get 
    $\delta W = \delta Y @ M$


"""

part1_q2 = r"""
**Your answer:**


No, not required, but very handy. Given some MLP we can compute the gradient of the parameters by hand and hardcode their calculations given a batch X,y. but that approach makes it hard to make changes to the architecture, and is not as friendly or modular.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.01, 0.05, 0.02
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    reg = 0.001
    lr_vanilla = 0.03
    lr_momentum = 0.005
    lr_rmsprop = 0.0003
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = (
        0.01,
        0.002,
    )
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. Compared to the models trained without dropout, I expected that the models trained with dropout will outperform the former on the test set while achieveing worse results on the traning set.<br>
In other words I expected the former model to overfit, and the later to generalize better compared.<br>
That didnt quite occur in my run.<br>
As we can see in the graphs the model trained without dropout did in fact overfit, but still got better results on the test set than the results achieved by the models trained with dropout.<br>
My guess as to the source of this behaviour is low learning rate or too little training epochs as we can see that the test accuracy of the dropout models are in a good trend compared to the model without dropout.
<br><br>
2. Comparing low dropout to high dropout, I expected to see better generalization from the high setting, and faster training from the low setting. Though its hard to guess exactly what setting will perform better.<br>
Again here, the graph doesnt reflect the behaviour I expected, as both setting perform quite similarly and its hard to tell them apart.<br>


"""

part2_q2 = r"""
**Your answer:**

Yes, it is possible.
The cross entropy loss is considering only the score the model gave to the correct class.<br>
We can imagen some situation in which out model is predicting correctly some label y1 (in the test set) with high confidence.<br>
We then perform gradient step with some batch, and it's possible that after the update, for the given label y1, we predict it with lower confidence and thus get higher loss while predicting the same.<br>
Though after this update it is also likely that the prediction of some other label y2 turned from a bad prediction to a correct one.<br>
Overall, in this (possible) scenario, after some optimizer step we get worse loss on the test set, while getting better accuracy.<br>
While I have described a scenario that happens over some batch step, we can see how this is possible even with a whole epoch using the same logic.<br>


"""

part2_q3 = r"""
**Your answer:**

1. Back propagation is the process by which we calculate the gradients of the loss with respect to the model parameters.<br>
Gradient descent is the process by which we use this gradient to update the model parameters to obtain a model with lower loss and thus better performace.<br>

2. In gradient descent (GD) we update the model based on the gradient of the loss with respect to the entire training set.<br>
This results in each step being clean (not noisy) and so we get rather consistent improvement in the loss.<br>
Though this method is hardly used because computing the gradient on the entire training set for each step is computationally not feasable mainly because, usually, the training set is too large to fit in memory.<br>
Stochastic Gradient Descent is similar in general with the difference of only calculating the gradient of the loss with respect to some (small) batch of training data. this results in noisy gradients, but is much faster for each step and turns out to get us faster optimization over all then regular GD.<br>

3. like we mentioned in (2) SGD is much faster than GD because in each step it only calculates the gradient with respect to loss of some small batch of data.<br>
Also because we are constantly changing the samples we compute the loss with, we get a dynamic error surface that is believed to help the optimizer get out of flat regions or sharp local minima since these features may disappear in the loss surface of subsequent batches.<br>

4. A. Yes, considering he divides each loss of some set by the number of total sets before summing the gradients, he should get exactly the same result.<br>
this is because the loss on the entire training set is the mean of losses of all the samples, and because the derivating operation is linear the sum of gradients will indeed be the gradient of the sum.<br>
B. could be that the previous batches werent cleared from memory after their subsequent use.<br>


"""

part2_q4 = r"""
**Your answer:**

1.A. In forward mode AD on a computational graph of a functions as described in the question we initialize grad of the input node to be 1 and for each subsequent node calculate its derivative with respect to the value from the node before it.<br>
We then achieve each node's grad by multipling this kept derivative with the grad of the node before it.<br>
While in the regular algorithm we keep all the grad fields of all the nodes, we can keep only the grad of the node which came before the node which we currently are computing the grad field for.<br>
We can thus calculate the grad fields in parallel to calculating the forward pass, keeping also just the last node's val field and overall achieveing space complexity of O(1).<br>
B. In backward mode AD we initialize grad = 1 in the output node, and for each previous node calculate the outputs derivative with respect to its own val.<br>
similarly to A - to optimize for space - we can keep only the grad of the node which comes after the node which we are computing the grad for.<br>
Though an optimization of the val field space in memory is impossible because we need the val fields for all the nodes to be stored before starting the backwards pass.<br>
so overall even after the optimization we get space complexity of O(n).<br><br>

2. Yes, we can maintain the grad field only for nodes which all their upstream / downstream (depending on forward or backward AD) grads have yet to be calculated.<br>
Though the complexity is not promised to be O(1) even in the forward AD suggested method.<br><br>

3. deep architectures have deep computation graphs and thus keeping for each node val, grad fields becomes expensive.<br>
using those techniques might be crucial to train those networks without running out of memory.<br>

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 2
    hidden_dims = 10
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.functional.cross_entropy
    lr, weight_decay, momentum = 0.001, 0.001, 0.99
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. Optimization error: not high. we can see our model got good accuracy on the training set.
2. Generalization error: a bit high. the test error keeps trending up throughout which suggests no overfitting. but even though we get high accuracy on the training set, the accuracy on the test set is quite lower.
3. Approximation error: not high. based on the decision boundry we can see the model approximates the underlying distribution quite well.

"""

part3_q2 = r"""
**Your answer:**

We remember how we generated the training and validation sets:<br>
We created data from two different distributions <br>
Distribution 1: make_moons turned by 10 degrees <br>
Distribution 2: make_moons turned by 50 degrees <br>
<br>
Then we split:<br>
X_train, X_valid, y_train, y_valid = train_test_split(X[:N_train, :], y[:N_train], test_size=1/3, shuffle=False)<br>
<br>
Which means overall we had 4000 from distribution 1 and 4000 from distribution 2.<br>
Now because X_train takes 5333 examples we get in it 4000 from the distribution 1 and 1333 from distribution 2.<br>
While the X_valid set takes 2667 examples from distribution 2.<br>
Now we see the validation set has more examples turned by more degrees. <br>
So the classifier trained on the training set, is used to seeing examples turned less.<br>
Testing the classifier on the validation set, from visualiaing the graphs, I suspect the classifier will have alot of mistakes of The False negative kind. (predict 0 for class 1 examples)<br>
"""

part3_q3 = r"""
**Your answer:**

You're training a binary classifier screening of a large cohort of patients for some disease, with the aim to detect the disease early, before any symptoms appear. You train the model on easy-to-obtain features, so screening each individual patient is simple and low-cost. In case the model classifies a patient as sick, she must then be sent to furhter testing in order to confirm the illness. Assume that these further tests are expensive and involve high-risk to the patient. Assume also that once diagnosed, a low-cost treatment exists.

You wish to screen as many people as possible at the lowest possible cost and loss of life. Would you still choose the same "optimal" point on the ROC curve as above? If not, how would you choose it? Answer these questions for two possible scenarios:

A person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis and can then be treated.
A person with the disease shows no clear symptoms and may die with high probability if not diagnosed early enough, either by your model or by the expensive test.
Explain your answers.



"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""