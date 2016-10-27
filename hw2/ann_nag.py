import numpy as np
from ann_util import sigmoid, sigprime
from numpy.random import rand
from util import calErr

def validate(X, Y, w1, w2, b1, b2) :
	yHat = []

	for ct1 in range(X.shape[1]) :
		x = X[:, ct1:ct1+1]
		y = Y[ct1:ct1+1, :]

		z1 = np.dot(w1, x) + b1
		a1 = sigmoid(z1)

		z2 = np.dot(w2, a1) + b2
		a2 = sigmoid(z2)

		yHat.append(a2)

	return calErr(np.squeeze(np.asarray(yHat), axis=2), Y)

def adam_update(m ,v, dx, eta) :
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8

	m = beta1*m + (1 - beta1)*dx
	v = beta2*v + (1 - beta2)*(dx**2)

	return -1.0*eta*m / (np.sqrt(v) + eps)

def nag_update(v, dx, eta) :
	mu = 0.9
	v_prev = v
	v = mu*v - eta*dx

	return -1.0*mu*v_prev + (1+mu)*v

def fit(X, Y, size, testX, testY, batSize = 100, eta = 1e-3, maxIter = 1000) :
	# Init 
	w1 = rand(size, X.shape[0])
	w2 = rand(1, size)
	b1 = rand(size, 1)
	b2 = rand(1, 1)
	a1 = np.zeros((size, 1))
	a2 = np.zeros((1, 1))

	w1_grad = np.zeros(w1.shape)
	w2_grad = np.zeros(w2.shape)
	b1_grad = np.zeros(b1.shape)
	b2_grad = np.zeros(b2.shape)

	# adam
	mw1 = np.zeros(w1.shape)
	mw2 = np.zeros(w2.shape)
	mb1 = np.zeros(b1.shape)
	mb2 = np.zeros(b2.shape)

	vw1 = np.zeros(w1.shape)
	vw2 = np.zeros(w2.shape)
	vb1 = np.zeros(b1.shape)
	vb2 = np.zeros(b2.shape)

	for ct1 in range(maxIter) :
		print("Iter: ", ct1)
		 
		trainErr = 0
		batCt = 0
		for ct2 in range(X.shape[1]) :
			batCt += 1
			x = X[:, ct2:ct2+1]
			y = Y[ct2:ct2+1, :]

			# forward
			z1 = np.dot(w1, x) + b1
			a1 = sigmoid(z1)

			z2 = np.dot(w2, a1) + b2
			a2 = sigmoid(z2)

			# back prop
			del2 = a2 - y
			del1 = np.dot(w2.T, del2) * sigprime(z1)

			b2_grad += del2
			w2_grad += np.dot(del2, a1.T)
			b1_grad += del1
			w1_grad += np.dot(del1, x.T)

			trainErr += np.abs(np.rint(a2) - y)


			#################### Simple SGD #################
			# m = 1
			# b2 -= eta/m*b2_grad
			# w2 -= eta/m*w2_grad
			# b1 -= eta/m*b1_grad
			# w1 -= eta/m*w1_grad
			#################### Simple SGD #################
			#################### ADAM #################
			# if (batCt == batSize) :
			# 	b2_grad /= batSize
			# 	w2_grad /= batSize
			# 	b1_grad /= batSize
			# 	w1_grad /= batSize
			# 	b2 += adam_update(mb2, vb2, b2_grad, eta)
			# 	w2 += adam_update(mw2, vw2, w2_grad, eta)
			# 	b1 += adam_update(mb1, vb1, b1_grad, eta)
			# 	w1 += adam_update(mw1, vw1, w1_grad, eta)
			# 	batCt = 0
			#################### ADAM #################
			if (batCt == batSize) :
				b2_grad /= batSize
				w2_grad /= batSize
				b1_grad /= batSize
				w1_grad /= batSize
				b2 += nag_update(vb2, b2_grad, eta)
				w2 += nag_update(vw2, w2_grad, eta)
				b1 += nag_update(vb1, b1_grad, eta)
				w1 += nag_update(vw1, w1_grad, eta)
				batCt = 0

		# Training error
		print("Training Err: ", trainErr/X.shape[1])

		# Validation error
		print ("Validation Err: ", validate(testX, testY, w1, w2, b1, b2))

	return [w1, w2, b1, b2]
