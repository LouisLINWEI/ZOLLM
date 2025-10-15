### Activation-Guided Zeroth-Order Optimization (AGZO)

#### 1. Background: The Low-Rank Zeroth-Order (LOZO) Gradient Estimator

The Low-Rank Zeroth-Order (LOZO) algorithm represents an advancement in memory-efficient fine-tuning by aligning the *rank* of the perturbation with the known low-rank structure of gradients in Large Language Models (LLMs). LOZO constructs a low-rank, isotropic perturbation $p_l$ for each weight matrix $w_l$ using two standard normal matrices, $U_l\in \mathbb R^{m_l\times r_l}$ and $V_l\in \mathbb R^{n_l\times r_l}$, where $r_l$ is the prescribed rank.

Perturbation and Estimator: The layer-wise perturbation is defined as: $ P_l = U_l V_l^T $ The corresponding LOZO gradient estimator for the full model weight set W is: $$ \hat{\nabla}_{W_l}L = \frac{L(W + \epsilon P) - L(W - \epsilon P)}{2\epsilon} \left( \frac{P_l}{r_l} \right) $$ where $P=\{P_l\}$ represents the set of all layer-wise perturbations and $\epsilon$ is the perturbation magnitude. While LOZO effectively constrains the perturbation's rank, its directional orientation remains purely random. This leaves a significant opportunity for improvement by guiding the perturbation towards subspaces more likely to contain the true gradient.

#### 2. The Activation-Guided (AGZO) Estimator

AGZO introduces a principled approach to guide the perturbation by leveraging the structural properties of the gradient computation. The core insight stems from the gradient formula for a linear layer: $\nabla W_l=(\nabla z_l)H_l^T$, which implies that the row space of the gradient $\nabla W_l$ is contained within the column space of the layer's activation matrix $H_l$.

By aligning the perturbation with the principal directions of the activation's column space, AGZO ensures a higher cosine similarity with the true gradient. To achieve this efficiently, AG-ZO approximates the top-1 left singular vector of $H_l$, denoted as $u_l$, which represents the principal direction of the gradient subspace.

##### Top-1 Singular Vector Approximation via Power Iteration

A full SVD on the activation matrix $H_l$ would be prohibitively expensive. Instead, AGZO employs power iteration—a fast, memory-efficient method—to approximate the principal direction $u_l$.

Power Iteration Procedure (for $k$ steps, typically k≤5):

- Initialization: Generate a random unit vector $q_0$.
- Iteration Step: The vector is alternately multiplied by $H_l^T$ and $H_l$, with normalization at each step to ensure numerical stability: 
  - a.  $y_k=H_l^Tq_k$ 
  - b.  $y_k=y_k/\|y_k\|_2$ 
  - c.  $y_k=H_ly_k$ 
  - d. $q_{k+1}=y_k/\|y_k\|_2$
- Result: The approximated principal direction is $u_l=q_k$.

##### Rank-1 Perturbation and Estimator

The rank-1 perturbation $P_l$ is then constructed using the guided basis vector $u_l$ and a new random vector $r_l\in\mathbb R^{m_l\times1}$: $P_l = r_lu_l^T$ This construction aligns the perturbation's row space with the principal direction of the activations. The final AGZO estimator is: $ \hat{\nabla}_{W_l}L = \frac{L(W + \epsilon P) - L(W - \epsilon P_)}{2\epsilon} P_l $

#### 3. Memory-Efficient AG-ZO Iteration Workflow

The AG-ZO algorithm is designed to integrate seamlessly with the memory-saving mechanisms of methods like MeZO and LOZO. This ensures that the computational benefits of guided perturbations are achieved without sacrificing the near-inference-level memory footprint.

##### Full Iteration Sequence

For each ZO iteration at weight state $W^t$:

1. Guided Basis Generation ($u_l$):
   - Perform a standard forward pass. A forward hook is used to intercept the activation matrix $H_l$ for each layer.
   - Immediately perform the $k$ steps of power iteration using Hl to compute the principal direction vector $u_l$.
   - Critically, $H_l$ is discarded after $u_l$ is computed. Only the small vector $u_l$ is stored for each layer, preventing memory accumulation.
2. Random Perturbation Sampling (Seed):
   - Sample a single random seed, $s^t$. This seed is used to deterministically regenerate the layer-wise random vectors $\{r_l\}$ in-place during the two-point estimation, employing the memory-saving technique from MeZO.
3. Two-Point Estimation:
   - Forward Evaluation (L+): For each layer $l$, perturb the weights $W^t_l=W^t_l+\epsilon P_l$. The guided perturbation $P_l$ is constructed and applied in-place by regenerating $r_l$ from the seed st and multiplying it by the stored vector $u_l$. Compute the total loss $L_+$.
   - Backward Evaluation (L−): Then, compute $W^t_l=W^t_l-\epsilon P_l $for each layer, again constructing $P_l$ in-place. Compute the total loss $L_−$.
   - Weight Reset: Restore the original weights $W^t_l=W^t_l+\epsilon P_l$.
4. Parameter Update:
   - Calculate the projected gradient scalar: $g_{proj}=(L_+−L_−)/2\epsilon$.
   - Perform the final update: $W^{t+1}_l=W^t_l-\eta g_{proj}P_l $. The perturbation matrix $P_l$ is once again reconstructed in-place using the stored $u_l$ and the seed $s^t$.

This process ensures that no large activation matrices ($H_l$) or perturbation matrices ($P_l$) are ever stored globally, thereby maintaining the minimal memory footprint characteristic of advanced zeroth-order methods.




+#### 4. Implementation Notes (2025.10.13)
+
+以下记录在现有 `zo-LLM` 框架中落地 AGZO 时的主要改动与理由，方便代码审阅与后续维护。
+
+- `zo-bench/run.py` line 67-75, 147-156, 618-625：
+  - 在 `OurArguments.trainer` 的可选项中新增 `agzo`，并引入 `agzo_power_iter_steps` 超参（默认 5）。
+  - 当 `trainer="agzo"` 时，将 power iteration 步数写入 `args.tag`，方便区分实验。
+
+- `zo-bench/trainer.py` line 131-207, 440-558, 716-1134：
+  - 初始化阶段增加 `self.agzo_enabled/self.agzo_hooks/self.agzo_u` 等结构，只在 `trainer=agzo` 时启用。
+  - 训练主循环新增 `agzo` 分支，并确保在 `zo_update` 判定里包含 `agzo`，保持学习率调度一致。
+  - `_register_agzo_hooks` 对所有 `nn.Linear` 注册 forward hook；`_agzo_power_iteration` 基于激活计算主方向 `u_l`，只缓存向量避免存储整块激活。
+  - `agzo_perturb_parameters` 使用现有随机种子 trick 构造 `r_l u_l^T` 扰动；若某层没有 `u_l`，保留 `zo_step` 作为回退并在日志中提示。
+  - `agzo_step` 按两点估计流程运行：先 `agzo_collect_basis` 更新 `u_l`，再对参数做 rank-1 扰动与恢复，最后逐层 `optimizer.step()`。
+
+- 其他注意事项：
+  - 激活 reshape 成 `(batch*seq_len, hidden)` 再做 power iteration，确保方向与权重行维度一致。
+  - Hook 中对激活 `detach()`，避免 Autograd 追踪带来的额外显存。
+  - 当前实现每个 step 都刷新 `u_l`；若需降低开销，可在未来扩展刷新频率或缓存策略。
+
+以上改动均以注释标明 “AGZO”，方便与原 MeZO 流程对照。


