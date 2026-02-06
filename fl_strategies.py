import torch
import copy

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = torch.load(model.state_dict()) # Clone to get structure if needed or just iterate
    # Actually safer to construct state dict
    new_state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(new_state_dict, strict=True)

def aggregate_fedavg(client_weights):
    """
    Standard FedAvg aggregation.
    client_weights: List of state_dicts from clients.
    Assumes equal weighting for simplicity or we can add weights based on data size.
    """
    avg_state = copy.deepcopy(client_weights[0])
    for key in avg_state.keys():
        for i in range(1, len(client_weights)):
            avg_state[key] += client_weights[i][key]
        avg_state[key] = torch.div(avg_state[key], len(client_weights))
    return avg_state

# FedProx is implemented in the Local Update (Client training loop), not just aggregation.
# But we can need a helper for the loss.
def proximal_term(local_model, global_model, mu):
    prox_loss = 0.0
    for w, w_t in zip(local_model.parameters(), global_model.parameters()):
        prox_loss += (w - w_t).norm(2)
    return (mu / 2) * prox_loss

# FedAdam Server Update
class FedAdamServer:
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.m = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.v = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self, global_model, client_weights):
        # 1. Compute Pseudo-Gradient (Average of client deltas)
        # Delta = Global - Client (or Client - Global depending on perspective)
        # Ideally: Average Client weights.
        # Update = Average_Client_Weights - Global_Weights
        
        avg_weights = aggregate_fedavg(client_weights)
        
        self.t += 1
        
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                grad = avg_weights[name] - param # The "gradient" direction towards average
                # Note: Usually Adam Minimiazes loss. Here we want to move TOWARDS avg_weights.
                # So if grad = avg - param, we should Add grad. 
                # FedAdam treats the difference as a gradient to MINIMIZE if we view it as loss.
                # Standard FedOpt: w_{t+1} = w_t + adam_update(delta)
                # Let's treat (avg - param) as the 'pseudo-gradient' g_t.
                # Actually proper FedOpt papers defined \Delta_t = \bar{w} - w_t.
                # w_{t+1} = w_t + \eta * Adam(\Delta_t)
                
                g_t = avg_weights[name] - param # This is the update step vector
                
                # Careful: Adam expects 'gradients' typically pointing DOWNHILL.
                # Here we want to MAXIMIZE/MOVE in direction of g_t.
                # or we just use the momentum logic directly.
                
                # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g_t
                
                # v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g_t ** 2)
                
                # Bias correction?? Usually yes.
                m_hat = self.m[name] # / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] # / (1 - self.beta2 ** self.t)
                
                # Update
                # w = w + lr * m / (sqrt(v) + eps)
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                param.add_(update)
        
        return global_model.state_dict()
