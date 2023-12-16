import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from utilities_1D import get_derivative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedContinuous:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, layers, t0, x0, t_lb, x_lb, t_ub, x_ub, t_f, x_f, u0,):
        """Construct a PhysicsInformedBar model"""

        self.t0 = t0
        self.x0 = x0
        self.t_lb = t_lb
        self.x_lb = x_lb
        self.t_ub = t_ub
        self.x_ub = x_ub
        self.t_f = t_f
        self.x_f = x_f
        self.u0 = u0
        self.model = self.build_model(layers[0], layers[1:-1], layers[-1])
        self.train_cost_history = []

    def build_model(self, input_dimension, hidden_dimension, output_dimension):
        """Build a neural network of given dimensions."""

        nonlinearity = torch.nn.Tanh()
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(nonlinearity)
        for i in range(len(hidden_dimension)-1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
            modules.append(nonlinearity)

        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))

        model = torch.nn.Sequential(*modules).to(device)
        print(model)
        print('model parameters on gpu:', next(model.parameters()).is_cuda)
        return model

    def u_nn(self, t, x):
        """Predict temperature at (t,x)."""

        u = self.model(torch.cat((t,x),1))
        return u

    def f_nn(self, t, x):
        """Compute differential equation."""

        u = self.u_nn(t, x)
        u_t = get_derivative(u, t, 1)
        u_xx = get_derivative(u, x, 2)
        u_xxxx = get_derivative(u, x, 4)
        
        f = u_t + 0.975 * u + u ** 3 + 2 * u_xx + u_xxxx
        return f

    
    def cost_function(self):
        """Compute cost function."""

        u0_pred = self.u_nn(self.t0, self.x0)
        u_lb_pred = self.u_nn(self.t_lb, self.x_lb)
        u_x_lb_pred1 = get_derivative(u_lb_pred, self.x_lb, 1)
        u_x_lb_pred3 = get_derivative(u_lb_pred, self.x_lb, 3)
        u_ub_pred = self.u_nn(self.t_ub, self.x_ub)
        u_x_ub_pred1 = get_derivative(u_ub_pred, self.x_ub, 1)
        u_x_ub_pred3 = get_derivative(u_ub_pred, self.x_ub, 3)
        f_pred = self.f_nn(self.t_f, self.x_f)

        mse_0 = torch.mean((self.u0 - u0_pred)**2)
        mse_b = 5*(torch.mean(u_x_lb_pred1**2) + torch.mean(u_x_ub_pred1**2))+(torch.mean(u_x_lb_pred3**2) + torch.mean(u_x_ub_pred3**2))+4*((torch.mean(u_x_lb_pred1)*torch.mean(u_x_lb_pred3))+(torch.mean(u_x_ub_pred1)*torch.mean(u_x_ub_pred3)))
        mse_f = torch.mean((5e-4*f_pred)**2)  # 5e-4 is a good value for balancing

        return mse_0, mse_b, mse_f

    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Select optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        ########################################################################
        elif optimizer=='L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters())

            def closure():
                self.optimizer.zero_grad()
                mse_0, mse_b, mse_f = self.cost_function()
                cost = mse_0 + mse_b + mse_f
                cost.backward(retain_graph=True)
                return cost
        ########################################################################

        # Training loop
        for epoch in range(epochs):
            mse_0, mse_b, mse_f = self.cost_function()
            cost = mse_0 + mse_b + mse_f
            self.train_cost_history.append([cost.cpu().detach(), mse_0.cpu().detach(), mse_b.cpu().detach(), mse_f.cpu().detach()])

            if optimizer=='Adam':
                # Set gradients to zero.
                self.optimizer.zero_grad()

                # Compute gradient (backwardpropagation)
                cost.backward(retain_graph=True)

                # Update parameters
                self.optimizer.step()

            ########################################################################
            elif optimizer=='L-BFGS':
                self.optimizer.step(closure)
            ########################################################################

            if epoch % 100 == 0:
                # print("Cost function: " + cost.detach().numpy())
                print(f'Epoch ({optimizer}): {epoch}, Cost: {cost.detach().cpu().numpy()}')

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        train_cost_history = np.asarray(self.train_cost_history, dtype=np.float32)

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        mse_0, mse_b, mse_f = ax.plot(train_cost_history[:,1:4])
        mse_0.set(color='r', linestyle='dashed', linewidth=2)
        mse_b.set(color='k', linestyle='dotted', linewidth=2)
        mse_f.set(color='silver', linewidth=2)
        plt.legend([mse_0, mse_b, mse_f], ['MSE_0', 'MSE_b', 'MSE_f'], loc='lower left')
        plt.tight_layout()
        plt.savefig('cost-function-history.eps')
        plt.show()
