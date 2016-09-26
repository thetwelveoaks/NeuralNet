import math
alpha = 0.8
a = 1
b = 0.5
w = [0, 0, 0, 0]
x1 = [-1, 0.8, 0.5, 0.0]
x2 = [-1, 0.9, 0.7, 0.3]
x3 = [-1, 1.0, 0.8, 0.5]
x4 = [-1, 0.0, 0.2, 0.3]
x5 = [-1, 0.2, 0.1, 1.3]
x6 = [-1, 0.2, 0.7, 0.8]
training_pairs = [[x1, 0], [x2, 0], [x3, 0], [x4, 1], [x5, 1], [x6, 1]]

def activate(synaptic_in):
    import math
    return 1 / (1 + math.exp(-synaptic_in))

def activate_derv(synaptic_in):
    return activate(synaptic_in) * (1 - activate(synaptic_in))

converged = False
epoch = 0
emax = 1e-3
eps = 1e-10
print_enabled = True
while(not converged):
    err = 0
        
    for pair in training_pairs:
        
        u = sum([wgt * ipt for wgt, ipt in zip(w, pair[0])])
        y = activate(u)
        err += (pair[1] - y) ** 2 / 2
        
        delta = (pair[1] - y) * activate_derv(u)
        w = [wgt + alpha * delta * ipt for wgt, ipt in zip(w, pair[0])]
        
    if err - emax < eps:
        converged = True
        
    epoch += 1
    

print("{}{}".format("ANN converges after Epoch ", epoch))
print("{}{}".format("Final Weights: ", w))
    
