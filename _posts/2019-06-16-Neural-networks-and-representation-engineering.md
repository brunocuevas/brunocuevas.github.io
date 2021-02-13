---
layout: post
title: NN and representation engineering
---

# Neural networks and representation engineering

In 2016, when I studied NNs for the first time, I was told by the proffesor that
NNs were just a new hyped trend that would eventually die as it did before
twice (in the 40s and 80s). Two years later, I am sure that that statement is
not longer true. NNs are reborn to stay.

But somehow parts of the community remain skeptical about it. In this post I'd
like to show something that does not seem very obvious when you deal with
neural networks: neural networks *engineer* representation. From a set of
values with given dimensions, NNs generate different representations along
the different layers that match the dimensions of the internal matrices, **until
they bring the problem into a representation where the problem is easier to
solve**.

Let's begin with some toy problem: a set of points in the plane. The points that
lie at a distance lower than a given treshold from two centers have class *1*,
and those outside, *0*. An easy problem.

	import torch
	import torch.nn as nn
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	plt.set_cmap('bwr')

	n_samples = 1000

	x = (np.random.rand(n_samples, 2) - 0.5) * 2.0  #*
	y = np.zeros(n_samples)
	for i in range(n_samples):
	    y[i] = (in_circle_1(x[i,:]) or in_circle_2(x[i,:]))

![img1](/assets/post_2019_06_16/fig1.png)

In this two dimensions, it would be difficult that any ML technique separates
the values of the classes. However, a simple NN can do it. For this example,
I'll use a PyTorch network with high capacity to illustrate my point.



	# Declare the NN

	space_map = nn.Sequential(
	    nn.Linear(2, 20),
	    nn.ReLU(),
	    nn.Linear(20, 5),
	    nn.ReLU(),
	    nn.Linear(5, 2),
	    nn.ReLU(),
	    nn.Linear(2, 1),
	    nn.Sigmoid()
	)
	xt = torch.tensor(x, dtype=torch.float)
	labels = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
	historic_loss_function = np.zeros(1000)
	optimizer = torch.optim.SGD(lr=1e-4, params=space_map.parameters())
	for i in range(1000):
	    output = space_map.forward(xt)
	    loss = -((labels * torch.log(output)) + (1 - labels) * torch.log(1 - output)).sum()
	    historic_loss_function[i] = loss.item()
	    loss.backward()
	    optimizer.step()
	    space_map.zero_grad()



This network usually does ok for this task. The training history has a strange
shape, like it has a step in the middle. It seems that it is a transition
point between a fit and unfit network. Those times where I didnt see that
step, the NN was not trained. Anyway, it's just an observation.

![img1](/assets/post_2019_06_16/fig2.png)

The results seems nice.

![img1](/assets/post_2019_06_16/fig3.png)

Now, let's prove my point. We are asking a NN to classify points according to
their probability of being of the class *1*. We have used for that a sigmoid
output unit, which means that those points that have in the preactivation layer
(the layer that is before the output unit) values larger than 0 will
be mostly 1, and those with values lower than 0, will be mostly 0. Let's take a
look:

![img1](/assets/post_2019_06_16/fig4.png)

You can observe now that the presigmoidal values are mostly ~3-4 for the
class 1, and the values for the class 0 lie in the range -25 to 0. So, the NN
modified the representation of the points until they were all placed in a
single dimension where they could be easily separated.

In the next two figures I'll represent in the Z-axis the value of the
pre-sigmoidal layer and the final output (post-sigmoidal) of the network. You
can see some shape of tableland that becomes more sharper after the sigmoid
is applied.

![img](/assets/post_2019_06_16/fig5.png)

![img](/assets/post_2019_06_16/fig6.png)

But this is not over. What happens in the layer before the presigmoidal? In
that layer we have and output of two dimensions, so we can easily plot it.

![img](/assets/post_2019_06_16/fig7.png)

Can you notice? The output of this layer already shows the separation of the
points belonging to each of the classes. This is an important feature of neural
networks: the inner layers of a network can provide important information
about the input, regardless of the specific output. This is used in what
the ML community calls *feature extraction*. Neural networks trained for
one purpose can easily be used to extract properties of the input in a second
ML method whose training is easier and less prone to overfitting.

I'm working in this idea right now to map chemical atoms in some sort of
ML chemical space that allows new kind of predictions. Wish me luck!
