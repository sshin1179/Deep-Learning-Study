# Attention Is All You Need

At its core, the self-attention mechanism revolves around the interplay of three components: **key**, **query**, and **value**. These are vital for understanding how information is weighted and propagated in attention models, such as the Transformer.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right) \cdot V $$

When $Q = K$, the term $QK^{T}$ captures the self-attention, indicating how similar elements within the matrix $Q$ are to one another.

## Why Use $\sqrt{d_k}$?

[proof.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5df62dc5-72a1-4bf6-ac20-b804fe52d000/proof.pdf)

### Lemma 1
It is on the assumption that the components of $q$ and $k$ are independent from random variables with mean 0 and variance 1. Their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_{i}k_{i}$ has mean 0 and variance $d_{k}$.

Here’s the proof. According to the *linearity of expectation*, the mean of the sum of random variables equals the expected value of $Z$:

$$ E[q \cdot k] = E\left[\sum_{i=1}^{d_k} q_i k_i\right] $$

By the ***linearity of expectation***:

$$ = \sum_{i=1}^{d_k} E[q_ik_i] $$

Another assumption is that random variables are i.i.d (independently identically distributed):

$$ = \sum_{i=1}^{d_k} E[q_i]E[k_i] = 0 $$

Thus, the mean of $q \cdot k$ equals 0. For variance:

$$ \text{var}[q \cdot k] = \text{var}\left[\sum_{i=1}^{d_k}q_kk_i\right] $$

Following the same logic:

$$ = \sum_{i=1}^{d_k}\text{var}[q_kk_i] = \sum_{i=1}^{d_k} \text{var}[q_i] \text{var}[k_i] = d_k $$

To make the dot product have a mean of 0 and standard deviation of 1, it's divided by $\sqrt{d_k}$. However, nowadays, this normalization is often omitted as normal distribution is not always assumed, especially when layer normalization is not used.

### Scaled Dot Product Attention

**Scaled Dot Product Attention** is the process of this calculation.

Given that **Query**, **Key**, and **Value** are all $3 \times 1$ matrices:

$$ 
Q = K = V = \begin{bmatrix} 
v_1 \\ 
v_2 \\ 
v_3 
\end{bmatrix} 
$$

Since $QK^{T}$ results in a $3 \times 3$ matrix:

$$ 
QK^T = \begin{bmatrix} 
v_1 \cdot v_1 & v_1 \cdot v_2 & v_1 \cdot v_3 \\ 
v_2 \cdot v_1 & v_2 \cdot v_2 & v_2 \cdot v_3 \\ 
v_3 \cdot v_1 & v_3 \cdot v_2 & v_3 \cdot v_3 
\end{bmatrix} 
$$

We then divide $QK^{T}$ by $\sqrt{d_k}$, obtaining the **attention weight**:

$$ 
\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) = \begin{bmatrix} 
w_{11} & w_{12} & w_{13} \\ 
w_{21} & w_{22} & w_{23} \\ 
w_{31} & w_{32} & w_{33} 
\end{bmatrix} 
$$

Given the value matrix, we calculate:

$$ 
\frac{QK^T}{\sqrt{d_k}} \times V = \begin{bmatrix} 
w_{11} & w_{12} & w_{13} \\ 
w_{21} & w_{22} & w_{23} \\ 
w_{31} & w_{32} & w_{33} 
\end{bmatrix} \times \begin{bmatrix} 
v_1 \\ 
v_2 \\ 
v_3 
\end{bmatrix} = \begin{bmatrix} 
y_1 \\ 
y_2 \\ 
y_3 
\end{bmatrix} 
$$

The attention mechanism gauges the similarity between a *query* (the word we're focusing on) and a *key* (the word we're comparing against). The resulting similarity scores are then used to weigh the importance of words in the **Value** matrix.

## List of Proofs

### **Linearity of Expectation**

Let $X$ and $Y$ be two random variables with $E[X]$ and $E[Y]$ as their expected values. The **Linearity of Expectation** states:

$$ E[X + Y] = E[X] + E[Y] $$

For discrete random variables, the proof is:

$$ E[X + Y] = \sum_{i} \sum_{j} (x_i + y_j) p_{X,Y}(x_i, y_j) $$

This expands to:

$$ = \sum_{i} \sum_{j} x_i p_{X,Y}(x_i, y_j) + \sum_{i} \sum_{j} y_j p_{X,Y}(x_i, y_j) $$

Simplifying further:

$$ = \sum_{i} x_i \sum_{j} p_{X,Y}(x_i, y_j) + \sum_{j} y_j \sum_{i} p_{X,Y}(x_i, y_j) $$

Which results in:

$$ = E[X] + E[Y] $$
