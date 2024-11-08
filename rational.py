# https://github.com/NBoulle/RationalNets/blob/master/src/PyTorch%20implementation/rational.py

"""

Implementation provided by Mario Casado (https://github.com/Lezcano)

# True for using Rational activation function,
# False for using ReLU
UseRational = True

class Net(torch.nn.Module):
    def __init__(self, UseRational):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1,50)
        if UseRational:
            self.R1 = Rational()
        else:
            self.R1 = F.relu
        self.fc2 = torch.nn.Linear(50,50)
        if UseRational:
            self.R2 = Rational()
        else:
            self.R2 = F.relu
        self.fc3 = torch.nn.Linear(50,1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x

"""

import torch

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to approximate ReLU.
    P(x) is now of degree 5 (6 coefficients) and Q(x) is of degree 4 (5 coefficients)
    """
    def __init__(self):
        super().__init__()
        # Increase coeffs tensor to handle higher orders
        # First column for P(x), second column for Q(x)
        self.coeffs = torch.nn.Parameter(torch.Tensor(6, 2))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with a simple pattern - you may want to tune these values
        self.coeffs.data = torch.Tensor([
            [1.1915, 0.0],    # x^5 term for P, x^4 term for Q
            [1.5957, 2.383],  # x^4 term for P, x^3 term for Q
            [0.5, 0.0],       # x^3 term for P, x^2 term for Q
            [0.0218, 1.0],    # x^2 term for P, x^1 term for Q
            [0.0, 0.0],       # x^1 term for P, x^0 term for Q
            [0.0, 0.0]        # x^0 term for P
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()  # Ensure highest order of Q is zero
        # Update exponents for higher order
        exp = torch.tensor([5., 4., 3., 2., 1., 0.], 
                          device=input.device, 
                          dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output