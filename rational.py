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
        self.coeffs = torch.nn.Parameter(torch.Tensor(6, 2))
        self.eps = 1e-7  # Small constant to prevent division by zero
        self.reset_parameters()

    def reset_parameters(self):
        # Even more conservative initialization
        # Start close to ReLU-like behavior with minimal higher-order terms
        self.coeffs.data = torch.Tensor([
            [0.01, 0.0],    # x^5 term for P, x^4 term for Q
            [0.02, 0.01],   # x^4 term for P, x^3 term for Q
            [0.1, 0.1],     # x^3 term for P, x^2 term for Q
            [1.0, 0.5],     # x^2 term for P, x^1 term for Q
            [1.0, 2.0],     # x^1 term for P, x^0 term for Q
            [0.0, 0.0]      # x^0 term for P
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()  # Ensure highest order of Q is zero
        
        # Clip input to prevent extremely large values
        input = torch.clamp(input, -100, 100)
        
        exp = torch.tensor([5., 4., 3., 2., 1., 0.], 
                          device=input.device, 
                          dtype=input.dtype)
        
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        
        # Add small epsilon to denominator to prevent division by zero
        denominator = PQ[..., 1] + self.eps
        
        # Ensure denominator is positive
        denominator = torch.abs(denominator)
        
        output = torch.div(PQ[..., 0], denominator)
        
        # Clip output to prevent explosion
        output = torch.clamp(output, -100, 100)
        
        return output