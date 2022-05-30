# minGPT

Andrej Kaparthy's minGPT model buit in PyTorch. Two types of attention are available for use in the model: a standard masked multi-headed self attention or a Synthesizer self attention. The attention classes are found in attention.py. 

Synthesizer attention: $$Y_{i} = softmax(ReLU(XA_{i} + b_{1})B_{i} + b_{2})(XV_{i})$$

The synthesizer variant eschews the pairwise dot products and directly computes the ℓ × ℓ matrix of attention scores by mapping each d-dimensional vector of each head
for X to an ℓ-dimesional vector of unnormalized attention weights.
