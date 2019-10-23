import torch

# Derivatives

# slope of a tangeant line
# rate of change in a function
# high rate of change = steep slope, high derivative
# low rate of change = flat slope, low derivative

# Leibniz' notation (aka differential notation)
# slope = dy / dx
# Legrange notation, using f-prime of x_1, f'(x1) 
# In physics, y-dot
# In math, y-prime

# For more depth:
# https://www.khanacademy.org/math/differential-calculus/dc-diff-intro/dc-diff-calc-intro/v/derivative-as-a-concept?modal=1


# rules
# The 'd' means "derivative of", as in 'dx' means "derivative of x"

# addition (f + g)' = f' + g'
# multiplication (f * g)' = (f * dg) + (g * df)
# powers (x^n)' = (d / dx)*x^n = n*x^n-1
# inverse (1 / x)' = -(1 / x^2)
# division (f / g)' = (df * (1/g)) + ((-1 / g^2) * dg * f)



# Backpropagation by auto-differentiation
# gradient = derivative, in context of neural networks

# chain rule

# intialize values
# requires_grad lets torch know we need to calculate derivatives.
x = torch.tensor(-3., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = torch.tensor(-2., requires_grad=True)

q = x + y
f = q * z

f.backward()

print("Gradient of z is: ", + str(z.grad))
print("Gradient of y is: ", + str(y.grad))
print("Gradient of x is: ", + str(x.grad))