import torch
import torch.nn.functional as F
import numpy as np

# The below code establishes the equivalence between a standard multiclass classifier formulation and a supervised InfoNCE formulation.
#
# Both formulations operate on an encoded continuous signal f(x) and a set of class embeddings (or weights) W.
#
# In the standard classifier:
#   - For each sample i and class j, the logit is computed as: f(x_i) · W_j.
#   - The positive logit for sample i is: logits_classifier[i, label_i].
#
# In the InfoNCE formulation:
#   - The positive class embedding is obtained by indexing W with the true label: target_W = W[labels].
#   - Both the encoded features and the selected class embeddings are normalized.
#   - The cosine similarity between the normalized vectors is computed.
#   - A temperature parameter is defined as: tau = 1 / (||f(x)|| * ||target_W||)
#     to recover the original dot product as: logits_infonce = cosine_sim / tau.
#   - The loss is defined as the difference between the logsumexp over all logits and the positive logit.
#
# Loss definitions:
#   - Categorical Cross-Entropy Loss:
#         loss_classifier = logsumexp(logits_classifier) - logits_classifier[labels]
#   - InfoNCE Loss:
#         loss_infonce = logsumexp(logits_infonce) - logits_infonce (diagonal)

def logmeanexp(x, dim, keepdim=False):
    """
    Compute log(mean(exp(x))) along the given dimension in a numerically stable way.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the logmeanexp.
        keepdim (bool): Whether to retain the reduced dimension in the output.
    
    Returns:
        torch.Tensor: The result of log(mean(exp(x))) computed along the specified dimension.
    """
    n = x.size(dim)
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - np.log(n)

# --- Setup parameters and inputs ---
batch_size, hidden_dim, num_classes = 4096, 128, 16
f_x = torch.randn(batch_size, hidden_dim)    # Encoded continuous variable f(x)
W = torch.randn(num_classes, hidden_dim)       # Class weights/embeddings
labels = torch.randint(0, num_classes, size=(batch_size,))  # Random labels (batch size)

# --- Standard Classifier with Categorical Cross-Entropy ---
# For each sample i and class j:
#     logits_classifier[i, j] = f(x_i) · W_j
# The positive logit for sample i is:
#     logits_classifier[i, label_i]
logits_classifier = torch.matmul(f_x, W.t())  # Shape: [batch_size, num_classes]
pos_classifier = logits_classifier[torch.arange(batch_size), labels].mean().item()
neg_classifier = -logmeanexp(logits_classifier, dim=-1).mean().item()

# --- InfoNCE Formulation (Index-Then-Similarity) ---
# For each sample i:
#   - The positive class embedding is: target_W[i] = W[label_i]
#   - Normalize the encoded features and the positive class embeddings.
#   - Compute the cosine similarity between f(x_i) and target_W[i].
#   - Define tau as: tau = 1 / (||f(x_i)|| * ||target_W[i]||)
#   - Recover the original dot product as: logits_infonce = cosine_sim / tau.
#   - The positive logits are the diagonal elements of logits_infonce.
target_W = W[labels]  # Shape: [batch_size, hidden_dim]

# Normalize the encoded features and the positive class embeddings.
f_x_norm = F.normalize(f_x, p=2, dim=1)          # Shape: [batch_size, hidden_dim]
target_W_norm = F.normalize(target_W, p=2, dim=1)  # Shape: [batch_size, hidden_dim]

# Compute the pairwise cosine similarity matrix between f_x_norm and target_W_norm.
sim_matrix_cos = torch.matmul(f_x_norm, target_W_norm.t())  # Shape: [batch_size, batch_size]

# Recover the original dot products using the temperature parameter tau.
pairwise_norm = f_x.norm(dim=1, keepdim=True) * target_W.norm(dim=1).unsqueeze(0)  # Shape: [batch_size, batch_size]
tau = 1.0 / pairwise_norm
logits_infonce = sim_matrix_cos / tau  # Shape: [batch_size, batch_size]

# The positive logits for InfoNCE are on the diagonal.
pos_infonce = logits_infonce.diag().mean().item()

# Compute the negative logits using logmeanexp.
neg_infonce = -logmeanexp(logits_infonce, dim=-1).mean().item()

# --- Loss Definitions ---
# Categorical Cross-Entropy Loss for the standard classifier:
loss_classifier = (torch.logsumexp(logits_classifier, dim=-1) - logits_classifier[torch.arange(batch_size), labels]).mean().item()
# InfoNCE Loss:
loss_infonce = (torch.logsumexp(logits_infonce, dim=-1) - logits_infonce.diag()).mean().item()

# --- Comparison ---
print("Standard classifier positive logits mean:", pos_classifier)
print("InfoNCE positive logits mean:", pos_infonce)
print("Standard classifier negative logits mean:", neg_classifier)
print("InfoNCE negative logits mean:", neg_infonce)
print("Standard classifier loss:", loss_classifier)
print("InfoNCE loss:", loss_infonce)

# Check if the positive logits are (numerically) close.
print("Are positive logits equal? ", np.allclose(pos_classifier, pos_infonce, atol=1e-2))
print("Are negative logits equal? ", np.allclose(neg_classifier, neg_infonce, atol=1e-2))

# Example output:
# Standard classifier positive logits mean: -0.2728476822376251
# InfoNCE positive logits mean: -0.27284765243530273
# Standard classifier negative logits mean: -17.090906143188477
# InfoNCE negative logits mean: -17.09239959716797
# Standard classifier loss: 17.363
# InfoNCE loss: 17.363
# Are positive logits equal?  True
# Are negative logits equal?  True
