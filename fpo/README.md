# Flow Matching Policy Gradients

This repo implements **Flow Policy Optimization (FPO)** for reinforcement learning in continuous action spaces.

Please see the [blog](https://flowreinforce.github.io/) and [paper](todo) for more details.

<table><tr><td>
    David&nbsp;McAllister<sup>1,*</sup>, Songwei&nbsp;Ge<sup>1,*</sup>, Brent&nbsp;Yi<sup>1,*</sup>, Chung&nbsp;Min&nbsp;Kim<sup>1</sup>, Ethan&nbsp;Weber<sup>1</sup>, Hongsuk&nbsp;Choi<sup>1</sup>, Haiwen&nbsp;Feng<sup>1,2</sup>, and Angjoo&nbsp;Kanazawa<sup>1</sup>.
    <strong>Flow Matching Policy Gradients.</strong>
    arXiV, 2025.
</td></tr>
</table>
<sup>1</sup><em>UC Berkeley</em>, <sup>2</sup><em>Max Planck Institute for Intelligent Systems</em>

## Updates

- **July 28, 2025:** Initial code release.

## Repository Structure

Our initial release contains two FPO implementations. Stay tuned for more updates!

### PHC

`phc/` contains PyTorch code for humanoid control experiments, which are based on the
[Puffer PHC](https://github.com/kywch/puffer-phc/tree/main).

## Acknowledgements

We thank Qiyang (Colin) Li, Oleg Rybkin, Lily Goli and Michael Psenka for helpful discussions and feedback on the manuscript. We thank Arthur Allshire, Tero Karras, Miika Aittala, Kevin Zakka and Seohong Park for insightful input and feedback on implementation details and the broader context of this work.
