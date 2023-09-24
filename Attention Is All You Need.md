# Attention Is All You Need

At its core, the self-attention mechanism revolves around the interplay of three components: **key**, **query**, and **value**. These are vital for understanding how information is weighted and propagated in attention models, such as the Transformer.

$$\text{Attention}(Q, K, V) = \text{softmax}\bigg(\frac{QK^{T}}{\sqrt{d_{k}}}\bigg) \cdot V \tag{1}$$

When $Q = K$, the term $QK^{T}$ captures the self-attention, indicating how similar elements within the matrix $*Q*$ are to one another.

## Why Use $\sqrt{d_k}$?

[proof.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5df62dc5-72a1-4bf6-ac20-b804fe52d000/proof.pdf)

**Lemma 1.** It is on the assumption that the components of $q$ and $k$ are independent from random variables with mean $0$ and variance $1$. Their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_{i}k_{i}$ has mean $0$ and variance $d_{k}$. 

Here’s the proof. According to the *linearity of expectation*, mean of sum of random variables equals to mean of $Z$. Therefore,

$$E[q \cdot k] = E\bigg[\sum_{i=1}^{d_k} q_i k_i\bigg]$$

According to the ***linearity of expectation***, mean of sum of random variables equals to expected value of $Z$. 

$$= \sum_{i=1}^{d_k} E[q_ik_i]$$

Another assumption is that random variables are i.i.d (independently identically distributed), which enables us to use the following:

$$= \sum_{i=1}^{d_k} E[q_i]E[k_i] = 0$$

That being said, mean of $q \cdot k$ equals to $0$. For variance, we may calculate the next step:

$$\text{var}[q \cdot k] = \text{var}\bigg[\sum_{i=1}^{d_k}q_kk_i\bigg]$$

We do the same logic here:

$$= \sum_{i=1}^{d_k}\text{var}\big[q_kk_i\big] = \sum_{i=1}^{d_k} \text{var}[q_i] \text{var}[k_i] = \sum_{i=1}^{d_k}1 = d_k$$

Therefore, the standard deviation is equal to $d_k$. To make it be mean of $0$ and standard deviation of $1$, it is divided into $\sqrt{d_k}$. **However, nowadays, it is not often being used, because normal distribution is not assumed for not using layer normalization at many times.**

### Scaled Dot Product Attention

**Scaled Dot Product Attention** is basically to do this calculation.

Given that **Query**, **Key**, and **Value** are equal to $3 \times 1$ matrix, 

$$Q=K=V=\begin{bmatrix} v_1 \\ v_2 \\ v_3\end{bmatrix}$$

Because **[Equation 1](https://www.notion.so/ca01db3970b64bf8ba11685b9e1a6b2c?pvs=21)** is to calculate $QK^{T}$, its shape becomes $3 \times 3$.

$$Q = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad
K^T = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$$

$$QK^T = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \begin{bmatrix} v_1^{T} & v_2^{T} & v_3^{T} \end{bmatrix} = \begin{bmatrix} v_1 \cdot v_1^{T} & v_1 \cdot v_2^{T} & v_1 \cdot v_3^{T} \\ v_2 \cdot v_1^{T} & v_2 \cdot v_2^{T} & v_2 \cdot v_3^{T} \\ v_3 \cdot v_1^{T} & v_3 \cdot v_2^{T} & v_3 \cdot v_3^{T} \end{bmatrix}$$

We divide $QK^{T}$ by $\sqrt{d_k}$, then we finally find out **attention weight**.

$$Softmax\bigg(\frac{QK^T}{\sqrt{d_k}}\bigg) = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \tag{2}$$

Given value is $3 \times 1$ matrix, we do the calculation again.

$$\frac{QK^T}{\sqrt{d_k}} \times V = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \times \begin{bmatrix} v_1 \\ v_2 \\ v_3\end{bmatrix} = \begin{bmatrix} y_1 \\ y_2 \\ y_3\end{bmatrix}\tag{3}$$

Think of the attention mechanism as gauging the similarity between a *query* (the word we're focusing on) and a *key* (the word we're comparing against). The resulting similarity scores are then used to weigh the importance of words in the **Value** matrix. The operation can be viewed as constructing an attention map that determines the significance of each word based on their similarities. The normalization with $\sqrt{d_k}$ was initially introduced to prevent the softmax function from having extremely small gradients with large dot product values. Nevertheless, this specific normalization may not always be utilized in modern applications.

### ***Linearity of Expectation***

Let $X$ and $Y$ be two random variables. The expectation (or expected value) of a random variable is denoted by $E[\cdot]$ . One of the fundamental properties of expectation is its linearity.

**Linearity of Expectation**:

$$
 E[X + Y] = E[X] + E[Y] 
$$

This property states that the expected value of the sum of two random variables is equal to the sum of their individual expected values, regardless of whether the two variables are dependent or independent.

**Proof**:

We can prove this for discrete random variables as follows:

Given:

$$
X \text{ takes on values } x_1, x_2, \ldots \\ Y \text{ takes on values } y_1, y_2, \ldots
$$

The joint probability mass function is $p_{X,Y}(x_i, y_j).$  Then, by definition: $E[X + Y] = \sum_{i} \sum_{j} (x_i + y_j) p_{X,Y}(x_i, y_j)$

Expanding the sum:

$$
 = \sum_{i} \sum_{j} x_i p_{X,Y}(x_i, y_j) + \sum_{i} \sum_{j} y_j p_{X,Y}(x_i, y_j) 
$$

This splits into two parts, and using the definition of expectation, we get:

$$
 = \sum_{i} x_i \sum_{j} p_{X,Y}(x_i, y_j) + \sum_{j} y_j \sum_{i} p_{X,Y}(x_i, y_j) 
$$

Which simplifies to:

$$
 = E[X] + E[Y] 
$$

Therefore, the linearity of expectation is proved for discrete random variables. A similar proof can be extended for continuous random variables using integrals.
