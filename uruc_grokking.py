"""
URUC Grokking Core - Reproducible continuous grokking in constrained dynamics
Author: Lucas (self-taught on iPhone XR, 2025-2026)
License: MIT

Core idea: V = |βC - e^{-H}|^p + λ_h H² + adaptive α push on H toward low entropy
Goal: sharp grokking onset (~1000 steps), low final manifold error, persistent CIC invariant (~0.115)

Latest Grok suggestions incorporated:
- H >= 0 hard cap
- adapt term scaled ×0.1
- p_power = 3.0 (softer amplification)
- RMSprop + momentum
- global gradient norm clipping
- mild weight decay on β/C
- low noise σ=0.005

Run in Colab or local Python 3.12+ (numpy, matplotlib)

Usage:
    python uruc_grokking.py
"""

import numpy as np
import matplotlib.pyplot as plt

def run_uruc_grokking(
    n_steps=30000,
    lr=0.005,
    lambda_h=0.01,
    alpha_adapt=1e-5 * 0.1,  # Grok scaled ×0.1
    p_power=3.0,             # Grok softened
    clip_norm=0.5,
    weight_decay=1e-4,
    noise_sigma=0.005,
    rms_alpha=0.99,
    momentum=0.9,
    save_plot=True,
    plot_filename='uruc_grokking_run.png'
):
    """
    Run one URUC grokking simulation.
    Returns histories and final metrics.
    """
    # Initial state
    beta = np.array([1.0])
    C    = np.array([1.0])
    H    = np.array([0.1])

    # RMSprop moments + momentum buffer
    v_beta = np.array([0.0])
    v_C    = np.array([0.0])
    v_H    = np.array([0.0])
    mom_H  = np.array([0.0])

    # Histories
    loss_hist = []
    acc_hist = []
    norm_hist = []
    energy_hist = []
    cic_hist = []
    h_hist = []
    eps_hist = []

    # ─── Main loop ──────────────────────────────────────────────────
    for step in range(n_steps):
        eps = beta[-1] * C[-1] - np.exp(-H[-1])
        v = np.abs(eps)**p_power + lambda_h * H[-1]**2

        # Gradients
        sign_eps = np.sign(eps) if eps != 0 else 0
        dV_dbeta = p_power * np.abs(eps)**(p_power-1) * sign_eps * C[-1]
        dV_dC    = p_power * np.abs(eps)**(p_power-1) * sign_eps * beta[-1]
        dV_dH    = p_power * np.abs(eps)**(p_power-1) * sign_eps * (-np.exp(-H[-1])) + 2 * lambda_h * H[-1]

        adapt = alpha_adapt * (H[-1] - 0.001)

        # Weight decay (L1 on β and C)
        dV_dbeta += weight_decay * np.sign(beta[-1])
        dV_dC    += weight_decay * np.sign(C[-1])

        # Global norm clipping
        grads = np.array([dV_dbeta, dV_dC, dV_dH + adapt])
        grad_norm = np.linalg.norm(grads)
        if grad_norm > clip_norm:
            grads *= clip_norm / grad_norm
        dV_dbeta, dV_dC, dV_dH_adapt = grads

        # RMSprop
        v_beta = rms_alpha * v_beta + (1 - rms_alpha) * dV_dbeta**2
        v_C    = rms_alpha * v_C    + (1 - rms_alpha) * dV_dC**2
        v_H    = rms_alpha * v_H    + (1 - rms_alpha) * dV_dH_adapt**2

        # Momentum on H
        mom_H = momentum * mom_H + (1 - momentum) * dV_dH_adapt

        # Updates
        beta_new = beta[-1] - lr * dV_dbeta / (np.sqrt(v_beta) + 1e-8)
        C_new    = C[-1]    - lr * dV_dC / (np.sqrt(v_C) + 1e-8)
        H_new    = H[-1]    - lr * mom_H / (np.sqrt(v_H) + 1e-8)

        # Hard H cap (Grok fix)
        H_new = max(0.0, H_new)

        # Noise
        beta_new += np.random.normal(0, noise_sigma)
        C_new    += np.random.normal(0, noise_sigma)
        H_new    += np.random.normal(0, noise_sigma)

        # Append
        beta = np.append(beta, beta_new)
        C = np.append(C, C_new)
        H = np.append(H, H_new)

        # Metrics
        loss_hist.append(v)
        acc = 1 - np.abs(eps) / max(1.0, np.abs(beta[-1]*C[-1]))
        acc_hist.append(acc)
        norm_hist.append(np.sqrt(beta[-1]**2 + C[-1]**2 + H[-1]**2))
        energy_hist.append(v)
        cic = np.log(np.abs(beta[-1])+1e-8) + np.log(np.abs(C[-1])+1e-8) + H[-1]
        cic_hist.append(cic)
        h_hist.append(H[-1])
        eps_hist.append(eps)

    # ─── Onset detection ────────────────────────────────────────────────
    loss_drop_thresh = 0.5
    window = 1000
    onset_step = -1
    for t in range(window, len(loss_hist)):
        if loss_hist[t] < (1 - loss_drop_thresh) * loss_hist[t - window]:
            onset_step = t
            break

    # Final metrics
    final_error = np.abs(beta[-1]*C[-1] - np.exp(-H[-1]))
    mean_cic = np.mean(cic_hist)
    cic_std = np.std(cic_hist)

    print(f"Grokking onset step: {onset_step if onset_step > 0 else 'Not detected'}")
    print(f"Final state: β = {beta[-1]:.4f}, C = {C[-1]:.4f}, H = {H[-1]:.4f}")
    print(f"Final manifold error: {final_error:.2e}")
    print(f"Mean CIC: {mean_cic:.6f} ± {cic_std:.6f}")

    # ─── Plots ──────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    axs[0,0].plot(loss_hist, 'b-', lw=1.2)
    axs[0,0].set_title('Loss (log)', fontsize=11)
    axs[0,0].set_yscale('log')
    if onset_step > 0:
        axs[0,0].axvline(onset_step, c='r', ls='--', lw=1.2, label='onset')
    axs[0,0].legend(fontsize=9)

    axs[0,1].plot(acc_hist, color='orange', lw=1.2)
    axs[0,1].set_title('Accuracy', fontsize=11)
    if onset_step > 0:
        axs[0,1].axvline(onset_step, c='r', ls='--', lw=1.2)

    axs[0,2].plot(norm_hist, 'g-', lw=1.2)
    axs[0,2].set_title('Weight Norm', fontsize=11)
    if onset_step > 0:
        axs[0,2].axvline(onset_step, c='r', ls='--', lw=1.2)

    axs[1,0].plot(energy_hist, 'purple', lw=1.2)
    axs[1,0].set_title('URUC Energy', fontsize=11)
    if onset_step > 0:
        axs[1,0].axvline(onset_step, c='r', ls='--', lw=1.2)

    axs[1,1].plot(cic_hist, 'cyan', lw=1.2)
    axs[1,1].set_title('CIC (target ~0.115)', fontsize=11)
    axs[1,1].axhline(0.115, c='k', ls='--', lw=0.8, label='target')
    if onset_step > 0:
        axs[1,1].axvline(onset_step, c='r', ls='--', lw=1.2)
    axs[1,1].legend(fontsize=9)

    axs[1,2].plot(h_hist, 'magenta', lw=1.2)
    axs[1,2].set_title('H value', fontsize=11)
    if onset_step > 0:
        axs[1,2].axvline(onset_step, c='r', ls='--', lw=1.2)

    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
    plt.show()

    return {
        'onset_step': onset_step,
        'final_error': final_error,
        'mean_cic': mean_cic,
        'cic_std': cic_std,
        'final_beta': beta[-1],
        'final_C': C[-1],
        'final_H': H[-1]
    }

# ─── Run example ────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_uruc_grokking()
    print("\nResults:", results)