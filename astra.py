import torch
from torch.optim import Optimizer


class Astra(Optimizer):
    """
    Custom PyTorch implementation of the Astra fixed-point iterative scheme.
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, beta=0.8, gamma=0.7, delta=0.6):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # alpha, beta, gamma, and delta (mapping to \zeta)
        defaults = dict(lr=lr, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        super(Astra, self).__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Astra optimizer requires a closure to evaluate gradients multiple times.")

        # ==========================================================
        # 1. First evaluation at \upsilon_n (Current weights)
        # ==========================================================
        loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            delta = group['delta']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # \upsilon_n
                v_n = p.data.clone()
                state['v_n'] = v_n

                # \phi(\upsilon_n) = \upsilon_n - lr * \nabla f(\upsilon_n)
                grad_v_n = p.grad.data.clone()
                phi_v_n = v_n - lr * grad_v_n
                state['phi_v_n'] = phi_v_n

                # z_n = (1 - \gamma_n) * \upsilon_n + \gamma_n * \phi(\upsilon_n)
                z_n = (1 - gamma) * v_n + gamma * phi_v_n

                # y_n = (1 - \beta_n) * \upsilon_n + \beta_n * z_n
                y_n = (1 - beta) * v_n + beta * z_n

                # k_n = (1 - \zeta_n) * \upsilon_n + \zeta_n * y_n
                k_n = (1 - delta) * v_n + delta * y_n
                state['k_n'] = k_n

                # Update weights to k_n to prepare for the second evaluation
                p.data.copy_(k_n)

        # ==========================================================
        # 2. Second evaluation at k_n
        # ==========================================================
        closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                k_n = state['k_n']
                phi_v_n = state['phi_v_n']

                # \phi(k_n) = k_n - lr * \nabla f(k_n)
                grad_k_n = p.grad.data.clone()
                phi_k_n = k_n - lr * grad_k_n

                # Final update: \upsilon_{n+1} = (1 - \alpha_n) * \phi(\upsilon_n) + \alpha_n * \phi(k_n)
                v_next = (1 - alpha) * phi_v_n + alpha * phi_k_n

                p.data.copy_(v_next)

        return loss

class AstraCollapsed(Optimizer):
    """Collapsed 2-parameter equivalent where c = gamma * beta * delta"""

    def __init__(self, params, lr=1e-2, alpha=0.9, c=0.336):
        defaults = dict(lr=lr, alpha=alpha, c=c)
        super(AstraCollapsed, self).__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Closure required")
        loss = closure()

        for group in self.param_groups:
            lr, c = group['lr'], group['c']
            for p in group['params']:
                if p.grad is None:
                    continue
                v_n = p.data.clone()
                phi_v_n = v_n - lr * p.grad.data.clone()
                self.state[p]['phi_v_n'] = phi_v_n
                k_n = (1 - c) * v_n + c * phi_v_n
                self.state[p]['k_n'] = k_n
                p.data.copy_(k_n)

        closure()

        for group in self.param_groups:
            lr, alpha = group['lr'], group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                k_n = self.state[p]['k_n']
                phi_v_n = self.state[p]['phi_v_n']
                phi_k_n = k_n - lr * p.grad.data.clone()
                p.data.copy_((1 - alpha) * phi_v_n + alpha * phi_k_n)

        return loss
