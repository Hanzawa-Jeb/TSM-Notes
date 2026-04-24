Methodology 长版记录
```latex
\section{Methodology}
To address the loss of plasticity during the online fine-tuning phase of Offline-to-Online (O2O) Reinforcement Learning (RL), we propose a unified framework, the \textbf{Teacher-Student Model (TSM)}. 
The transition from static offline datasets to dynamic online interactions often induces significant distribution shifts, causing standard algorithms to suffer from performance degradation and reduced adaptability in the fine-tuning stage. 
Our TSM mitigates this issue through two synergistic mechanisms: a plasticity-preserving distillation strategy to transfer knowledge to a freshly initialized online network, and a progressive normalization architecture to stabilize this transfer and the subsequent online adaptation.

\paragraph{Plasticity-Preserving Knowledge Distillation}

\begin{figure}[h]
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
  \caption{The figure shows the process of the knowledge distillation within the TSM framework.}
\end{figure}

Prior experiments~\ref{plasticity_toy} demonstrate that prolonged gradient-based optimization progressively degrades neural network plasticity, impairing the model's capacity to adapt to novel data distributions. 
This phenomenon is particularly pronounced during online fine-tuning, where distributional shift demands continuous representation updates. 
Consequently, the loss of plasticity directly manifests as suboptimal asymptotic performance in the fine-tuning phase.

Our approach is grounded in two fundamental premises. First, drawing on the principles of knowledge distillation, we posit that a student policy can effectively approximate the decision behavior of a teacher network even with constrained capacity. Second, empirical studies in deep RL consistently demonstrate that freshly initialized networks exhibit significantly higher plasticity than those subjected to prolonged optimization. 
Leveraging these insights, the first component of our TSM introduces a distillation phase bridging the offline and online stages. In this phase, we distill the behavioral prior from the offline actor into a newly initialized online actor. Throughout the following text, we denote the offline (teacher) actor as $\pi_{\text{offline}}$ and the online (student) actor as $\pi_{\text{online}}$.

The distillation objective is formulated as a mean squared error (MSE) loss over the offline dataset:
\begin{equation}
    \mathcal{L}_{\text{distill}}(\theta_{\text{train}}) = \mathbb{E}_{s \sim \mathcal{D}_{\text{off}}} \left[ \left\| \pi_{\text{online}}(s; \theta_{\text{train}}, \theta_{\text{freeze}}) - \pi_{\text{offline}}(s) \right\|_2^2 \right],
    \label{eq:distill_mse}
\end{equation}
where $\mathcal{D}_{\text{off}}$ denotes the offline dataset, and $\theta_{\text{online}} = \theta_{\text{train}} \cup \theta_{\text{freeze}}$ represents the partitioned parameter set of the student policy.

Specifically, during the distillation phase, we optimize only the trainable subset $\theta_{\text{train}}$ to align the student's action outputs with the teacher's policy via Eq.~\eqref{eq:distill_mse}. The remaining parameters $\theta_{\text{freeze}}$ are strictly maintained at their random initialization, ensuring their gradients remain zero throughout this phase. This selective update strategy preserves the inherent plasticity of the frozen components while transferring essential behavioral knowledge from $\pi_{\text{offline}}$.

\paragraph{Progressive Normalization for Stable Transfer}
While the aforementioned selective update strategy effectively preserves plasticity, the distillation process itself requires careful architectural regularization. Layer Normalization (LayerNorm) has consistently demonstrated strong efficacy across diverse machine learning paradigms, from large language models to representation learning. 
However, our empirical analysis reveals that naive application of standard LayerNorm within this distillation framework introduces notable optimization instabilities~\ref{exp}. 
To ensure the stability of the knowledge transfer and the subsequent online fine-tuning, the second component of our TSM integrates \textbf{CrescLayerNorm}, a tailored normalization mechanism embedded directly into the student network's architecture.

CrescLayerNorm preserves the core advantages of conventional LayerNorm---namely, training stabilization and output-scale regularization---while explicitly suppressing the gradient fluctuations that typically destabilize the knowledge distillation process. 
Specifically, we insert a CrescLayerNorm module after each fully connected layer, formulated as:
\begin{equation}
    \operatorname{CrescLayerNorm}(x; \lambda) = \lambda \cdot \operatorname{LayerNorm}(x) + (1 - \lambda) \cdot x,
    \label{eq:cresc_ln}
\end{equation}
where $x$ denotes the layer input and $\lambda \in [0, 1]$ controls the interpolation strength between full normalization and the raw identity mapping.

During the transition from offline distillation to online fine-tuning, $\lambda$ is dynamically scheduled to modulate the normalization effect. Specifically, $\lambda(t)$ increases linearly from $0$ to $1$ over a warm-up period at the onset of the online stage:
\begin{equation}
    \lambda(t) = \min \left( 1, \max \left( 0, \frac{t - T_{\text{off}}}{T_{\text{warmup}}} \right) \right),
    \label{eq:lambda_sched}
\end{equation}
where $t$ denotes the current training step, $T_{\text{off}}$ marks the end of the offline distillation phase, and $T_{\text{warmup}}$ specifies the duration of the scheduling window.

This scheduling design seamlessly complements the distillation objective. Initially, $\lambda \approx 0$ yields a near-identity mapping, allowing the student network to absorb the teacher's behavioral prior without the representational constraints imposed by strict normalization. As online fine-tuning commences, $\lambda$ progressively approaches $1$, gradually introducing LayerNorm's stabilizing dynamics. This smooth architectural transition enables the unified TSM framework to first capture fine-grained distributional details from the offline data, and subsequently leverage normalized optimization to enhance plasticity and maintain training stability under distributional shift. The detailed analysis of these results can be found in Section~\ref{exp}.
```