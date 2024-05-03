import torch

#Fix the seed for reproducibility
torch.manual_seed(12)

N_forward_passes, classes, h, w = 4, 16, 10, 20
p = torch.randn(N_forward_passes, classes, h, w) #4 samples, 16 channels, 10 rows, 20 columns

var_torch = torch.var(p, dim = 0, unbiased = False) #Usar aqui el unbiased = False!!! (Divide para N en lugar de N-1)

#Lo que teniamos antes devuelve una matrix de covarianza de dimensiones(h, w, classes, classes).
#Sumar los elementos de la diagonal de la matriz de covarianza (la traza) es igual a calcular torch.var, con unbiased=False
p_hat = torch.mean(p, dim = 0)
for n in range(p.shape[0]):
    #Epistemic term
    v_e = torch.permute((p[n, :, :, :] - p_hat), (1, 2, 0))
    term_1 = torch.unsqueeze(v_e, axis = 3)
    term_2 = torch.unsqueeze(v_e, axis = 2)
    epist_n = torch.matmul(term_1, term_2)
    if n == 0:
        total_epist_var = epist_n
    else:
        total_epist_var += epist_n

var_my = total_epist_var / N_forward_passes
var_my = torch.diagonal(var_my, dim1 = 2, dim2 = 3).permute(2, 0, 1)

equal = torch.allclose(var_torch, var_my)
print(var_torch[0, 0, 0], var_my[0, 0, 0])
print(equal)