Yes — a Gaussian-Process lets you turn the constant buffer

\theta into a state-dependent band \;\theta(x) that widens when your model is uncertain and tightens when it is confident.
Below is the statistical construction, ending in code you can drop into the same time-blocked CV scaffold you already have.

⸻

1 Frame the problem as a heteroscedastic regression

Instead of modelling class directly we model the signed deviation

d_i \;=\; \text{exec}_i - \text{mid}_i

as a Gaussian-Process over a feature vector x_i that captures today’s “state”:
	•	product ID (one-hot or a learned embedding)
	•	recent realised volatility, spread, order-book depth
	•	time-of-day, day-of-week
	•	anything you believe drives the price location

GP assumption For any new trade with state x

d(x)\;\sim\; \mathcal N\!\bigl(\;\mu(x),\;\sigma^2(x)\bigr).

Why a GP?
	•	Non-parametric μ(x) and σ(x) adapt to wildly different OTC instruments.
	•	Closed-form posterior The predictive variance σ²(x) tells you how sure the model is right now.
	•	Natural abstention rule A single cost table turns (μ, σ) into a decision boundary.

⸻

2 Derive the optimal state-dependent bands

Fix this symmetric cost matrix

decision a	true buy (y=+1)	true sell (y=-1)
buy (+1)	0	c_{\text{fp}}
sell (−1)	c_{\text{fn}}	0
skip (0)	c_{\text{skip}}	c_{\text{skip}}

With predictive mean μ and st-dev σ at state x the Bayes risk of each action is:

\begin{aligned}
R_{+1}(x) &= c_{\text{fp}}\; \Phi\!\bigl(-z\bigr)\\
R_{-1}(x) &= c_{\text{fn}}\; \Phi\!\bigl(z\bigr)\\
R_{0}(x)  &= c_{\text{skip}}
\end{aligned}
\quad\text{where } z=\mu(x)/\sigma(x).

Set R_{0}=R_{+1} and R_{0}=R_{-1} to obtain two z-score cut-offs

k_{+}= \Phi^{-1}\!\Bigl(1-\frac{c_{\text{skip}}}{c_{\text{fp}}}\Bigr),
\qquad
k_{-}= \Phi^{-1}\!\Bigl(\frac{c_{\text{skip}}}{c_{\text{fn}}}\Bigr).

Because costs are symmetric in many desks (c_\text{fp}=c_\text{fn}), k_{+}=k_{-}=k and the optimal state- dependent buffer is

\boxed{\;\theta(x)=k\,\sigma(x)\;}

so the rule becomes

 if     μ(x) >  +k σ(x)   ⇒   BUY
 elif   μ(x) <  -k σ(x)   ⇒   SELL
 else                     ⇒   SKIP

If you want different penalties for false-buys vs false-sells just keep k_{+}\neq k_{-}.

⸻

3 Time-blocked cross-validation to find k

The scalar k is the only hyper-parameter left.
Search it on a grid exactly as you already did, but:
	1.	Inside each fold
	•	fit the GP on the training block (training only sees its past),
	•	predict \mu, \sigma on the hold-out block,
	•	apply the rule with each candidate k,
	•	compute mean Bayes loss.
	2.	Pick the k that minimises the average loss across folds.

Because \sigma(x) already scales the buffer, k rarely lands outside the 0.5 – 3.0 range.

⸻

4 Code skeleton (GPyTorch, blocked CV)

import gpytorch, torch
from sklearn.model_selection import GroupKFold
from scipy.stats import norm
import numpy as np

def bayes_loss(mu, sigma, y, k, c_fp, c_fn, c_skip):
    buy  =  (mu >  k*sigma)
    sell =  (mu < -k*sigma)
    skip = ~(buy | sell)

    loss = np.zeros_like(mu)
    loss[ buy & (y==-1)]  = c_fp
    loss[sell & (y==+1)]  = c_fn
    loss[ skip            ] = c_skip
    return loss.mean()

def cv_search_k(grid_k, X, d, y, blocks,
                c_fp=1.0, c_fn=1.0, c_skip=0.1,
                n_splits=5):

    gkf = GroupKFold(n_splits)
    mean_loss = np.zeros_like(grid_k, dtype=float)

    for tr, te in gkf.split(X, groups=blocks):

        # 1) fit GP regression on training block
        train_x = torch.from_numpy(X[tr]).float()
        train_y = torch.from_numpy(d[tr]).float()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = gpytorch.models.ExactGP(train_x, train_y, likelihood,
                                        kernel=gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.RBFKernel()))
        model.train(); likelihood.train()

        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(200):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward(); opt.step()

        # 2) posterior on test block
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(X[te]).float()
            pred   = likelihood(model(test_x))
            mu     = pred.mean.numpy()
            sigma  = pred.variance.sqrt().numpy()

        # 3) evaluate each k
        for i, k in enumerate(grid_k):
            mean_loss[i] += bayes_loss(mu, sigma, y[te], k,
                                       c_fp, c_fn, c_skip)

    return grid_k[np.argmin(mean_loss / n_splits)]

Replace the kernel or use an additive kernel if you have separate “time” and “product” components.

⸻

5 Deployment loop
	1.	Nightly / hourly
	•	retrain the GP on the most recent N days,
	•	search k^\star with an expanding-window split,
	•	log (μ, σ) calibration diagnostics.
	2.	Live
	•	build x for the incoming OTC trade,
	•	get (μ, σ) from the GP’s predictive posterior,
	•	act via the rule with k^\star.

The buffer automatically grows in illiquid hours (σ up) and shrinks when the model is confident (σ down).

⸻

6 When a GP is overkill
	•	Millions of training trades → use a sparse GP or random-feature kernel regression instead.
	•	Only a few features and linear structure suffices → a Bayesian linear regression with heteroscedastic noise gives the same μ, σ formula faster.
	•	Stationarity across states holds after σ-scaling → your earlier constant-θ model is already near-optimal.

⸻

Take-away

A Gaussian-Process doesn’t replace the buffer; it computes a buffer that adapts to the current state via \theta(x)=k\,\sigma(x).
You still choose the cost ratios; the GP supplies μ and σ so the Bayes-optimal decision follows directly.