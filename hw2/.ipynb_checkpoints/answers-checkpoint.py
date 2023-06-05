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

1. Compared to the models trained without dropout, we expected that the models trained with dropout will outperform the former on the test set while achieveing worse results on the traning set.<br>
In other words we expected the former model to overfit, and the later to generalize better compared.<br>
As we can see from the graphs the configurations with dropout [0.4,0.8] classified the test set with 30 and 26 precent accuracy, compared to only 18 precent achieved by the configuration without dropout. All this while the configurations with dropout did worse on the training set than the configuration without dropout (around 95 without dropout compared to around 65 for both configurations with dropout).
Overall the configuration without dropout achieved very good results on the training set while performing poorly on the test set which suggest overfitting. And the configurations with dropout performed worse on the training set while achieving better results on the test set which suggests those configurations are much better in generalizing, as we expected.
<br><br>
2. Comparing low dropout to high dropout, we expected to see better generalization from the high setting, and faster training from the low setting.<br>
In our results however, the graph doesnt reflect the behaviour we expected, as both setting perform quite similarly on the training set and its hard to tell them apart. While in the final epochs the test score of the 0.4 dropout configuration beat the 0.8 configuration on the test set with accuracy better than the former by 4 precent.<br>



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
2. Generalization error: not high. the test error keeps trending up throughout which suggests no overfitting. We got good performace on the training set while getting just a little worse performance on the test set, which is to be expected.
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

First scenario: A person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis and can then be treated.<br>
In this case we will care more about not diagnosing a healthy patient as sick. Thats because the tests afterwards are expensive, and if he is indeed sick and we diagnosed him wrong, the non-lethal symptoms will appear and he will get help then.<br>
To conclude scenario 1, we would like to choose a point on the roc curve which corresponds to lower false positives then false negatives.<br>
<br>
Second scenario: A person with the disease shows no clear symptoms and may die with high probability if not diagnosed early enough, either by your model or by the expensive test.<br>
In this case, since there is a large chance for an infected person to die, we will care much more about having less false negatives. Thats because each false negative is a person who is sick but does not get treated, which puts him at high risk of dying. To compare, a false positive will only cost us in further testing money, but not in life.<br>
To conclude scenatio 2, we would like to choose a point on the roc curve which corresponds to lower false negatives the false positives.<br>
<br>


"""


part3_q4 = r"""
**Your answer:**

1. As the width increased, the model had more parameters and increased representational capacity. This led to decision boundaries that became more complex and flexible. The model performance improved with increasing width, as it could capture more intricate patterns in the data. However, at a certain point, further increasing the width did not significantly enhance performance and may have led to overfitting.<br>
<br>
2. As the depth increased, the model had more layers and increased the ability to capture hierarchical representations of the data. Model performance improved with increasing depth, as the model could capture more abstract features and interactions between variables. However, similar to the columns analysis, there was a point where further increasing depth did not result in significant performance gains and may have increased computational complexity.<br>
<br>
3. I would have expected the configuration with depth=1 and width=32 is likely to have worse performance because, while having the same number of parameters as the configuration with depth=4 and width=8, it is shallow and missing the notion of depth which usually helps the neural net be more expressive.<br>
The actual result though, was that the configuration with depth=1 and width=32 did better.<br>
I assume that is because our data is only 2 dimentional and does not require complex feature extraction.<br>
<br>
4. The experiment did not include evaluation on the test set without selecting thresholds.<br>
I would assume it did contribute to the performance on the test set, as we saw it improves results on the validation set in the previous sections of the notebook.<br>
<br>
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
    loss_fn = torch.nn.functional.cross_entropy
    lr, weight_decay, momentum = 0.01, 0.001, 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. Lets count the number of parameters for each block as described:<br>
<br>
Regular residual block:<br>
2 layers of 256(kernels) * (256*3*3 + 1)(parameters / kernel)<br>
so overall 2*256*(256*3*3 + 1) = 1,180,160 parameters.<br>
<br>
Bottleneck residual block:<br>
first layer:  64(kernels)  * (256*1*1 + 1)(parameters / kernel)<br>
second layer: 64(kernels)  * (64*3*3  + 1)(parameters / kernel)<br>
third layer:  256(kernels) * (64*1*1  + 1)(parameters / kernel)<br>
overall 70,016 parameters<br>
<br>
2. Number of floating point operations (assuming 256 x N x N input).<br>
<br>
Regular residual block:<br> 
first layer: 256(kernels) * N^2(inner products / kernel) * 256*3*3(floating point ops / inner product)<br>
second layer the same numbers, so overall about 1,179,648 * N^2 floating point operations.<br>
<br>
Bottleneck residual block:<br>
first layer: 64(kernels) * N^2(inner products / kernel) * 256*1*1(floating point ops / inner product)<br>
second layer: 64(kernels) * N^2(inner products / kernel) * 64*3*3(floating point ops / inner product)<br>
third layer: 256(kernels) * N^2(inner products / kernel) * 64*1*1(floating point ops / inner product)<br>
overall about 69,632 * N^2 floating point operations.<br>
<br>
3. Ability to combine the input:<br>
(1) Spatialy: In the regular residual block, we perform 2 3x3 convoluvtions compared to 1 in the bottleneck block, and so we get more spatial spread of information in the regular block. Though that is not inherent to the bottleneck block as we could have added another 3x3 convolution in the 64 channel domain while still performing much less operations than the regular residual block.<br>
(2) Across feature maps: We get the same spread across feature maps. Thats because all feature maps from some layer gets information from all feature maps of the previous layer. And that is true in both the regular redsidual block and the bottleneck residual block.<br>



"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1. In addition to the fact that training the network of depth 16 failed (for some obscure reason), from this experiment I see no immediate conclusion about the benefit of deeper networks. As the shallower models of depths 2,4 did better than the models of depth 8, and the deepest network of depth 16 failed to train.
2. As mentioned the models of depth 16 werent trainable. What may cause this are problems such as diminishing or exloding gradients (though from our graph we suspect vanishing gradients). To resolve this we have the well known solution which is residual connections. another possible solution is being careful about choosing the initialization of the weights.


"""

part5_q2 = r"""
**Your answer:**

We notice that the models with wider channels preformed slightly better than the ones with narrower channels, but not by much.
The performance of the models is similar (unnoticably better or worse) from the first experiment, which is not suprising as we ran in this experiment clones of models which ran in the previous experiment and the channels width did not make much of a difference.
"""

part5_q3 = r"""
**Your answer:**

In this experiment K = [64,128] fixed, with L changes in each run and gets values 2,3,4.
Seems reasonable to expect that the deeper models corresponding to L = 3 or 4 will perform better on the test set after training. 
Strangly, the most shallow model with L = 2 performed the best. with L = 3 and L = 4 performing similarly to each other and slightly worse than L = 2.

"""

part5_q4 = r"""
**Your answer:**

In this experiment we ran the following models.<br>
K=[32] fixed with L=8,16,32 varying per run.<br>
K=[64, 128, 256] fixed with L=2,4,8 varying per run.<br>
This time all the models were of the ResNet type.<br>
Performance-wise we notice that all models achieved test set accuracy of between 50 to 55 precent, similar to previous results from previous experiments - that is all except one model L8_K32 (resnet) which performed the best (of all experiments) at a test accuracy of 60.
We also notice without doubt the benefit of resnet to training large models.
We remember that in the first experiment we could not train a model of depth 16.
In this experiment we successfuly trained models of depth 24 (8*[64,128,256]) thus reassuring the known result that residual connections help train large models.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""

The model performed poorly.
On the dolphin image - it enclosed 2 of the 3 dolphins in bounding boxes just fine but classified them as persons, while the third dolphin was not even bounded properly and classifed wrong again, this time as a surfboard.
On the dogs image - it enclosed all the dogs in a bounding box just fine, but classified 2 of the 3 as cats instead. while not bounding the cat in the picture within a box.
possible reasons for the model failures are Insufficient training data, Class imbalance.
possible solution incorporating a larger and more diverse dataset.

"""


part6_q2 = r"""
**Your answer:**



"""


part6_q3 = r"""
**Your answer:**

first image:<br>
The first picture is an example of a failed classification due to bias. The model classifies the dog walking like a person as a person, because its not used to see dogs walking on 2. The models has a bias that dogs usually are not up straight.<br>
<br>
second image:<br>
this image is an example of bad lighting condition, due to the fog, the model classifies a car as a boat.<br>
<br>
third image:<br>
The model fails to detect the person in the left bottom of the picture. We suspect this stems mainly from the fact the person is partially occluded, and thus missing important features. Though it also may have to do with the fact that he is out of focus, anyway that is the best example of failure due to occlusion we could find.


"""

part6_bonus = r"""
**Your answer:**



"""