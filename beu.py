Below is a practical way to turn only mid-price, implied-vol and greeks into an estimate of the bid–ask spread for a 1-month ATM straddle, then compare that spread with the straddle’s typical 30-minute price fluctuations.

⸻

1 .  Fix the key Black–Scholes identities for an ATM straddle

For zero rates and dividends (adjust if they matter):

quantity	closed–form (ATM, BS)	numerical constant
Mid price P	C_{\text{ATM}}+P_{\text{ATM}}\;=\;2\,S\,\phi(d_1)\,\sigma\sqrt{T}	P\approx0.7979\,S\,\sigma\sqrt{T}
Vega V	2\,S\,\phi(d_1)\sqrt{T}	V\approx0.7979\,S\sqrt{T}
therefore	P = V\;\sigma	always

So with just the mid straddle price you already know vega: V=P/\sigma.

⸻

2 .  How far can the straddle move in 30 minutes?

Price changes in a delta-neutral straddle come from vega risk (volatility moves) and gamma risk (second-order spot moves).

component	driver	30-min variance of value
Vega term	V\,\Delta\sigma	V^{2}\;\sigma_{\sigma}^{2}\;\Delta t
Gamma term	\(\tfrac12\Gamma(\Delta S)^{2}\)	\(\tfrac14\Gamma^{2}S^{4}\sigma_{S}^{4}\;\Delta t^{2}\)

	•	\sigma_{\sigma}  = annual vol-of-vol (st-dev of implied‐vol moves).
	•	\sigma_{S}  = annual realised spot vol (use historical intraday).
	•	\Delta t = 30\text{ min} / (365\times24\times60) \approx 3.05\times10^{-4}\text{ yr}.

Because \Delta t is tiny, the vega linearly dominates unless you are pricing very large‐gamma products (equity index straddles usually aren’t).  A safe rule of thumb is:

\operatorname{SD}\big(\Delta P_{30\text{m}}\big)\;\approx\;V\;\sigma_{\sigma}\;\sqrt{\Delta t}.

⸻

3 .  Turn that variability into an economic spread

Market-makers usually quote a spread wide enough that the expected adverse move over their holding period is some multiple z of half the spread.  Set

\text{Half-spread }h  \;=\; z \; V \; \sigma_{\sigma}\;\sqrt{\Delta t}.

Typical choices:
	•	z = 1 for a “one-sigma” protection.
	•	Add inventory / funding buffers (often +5–10 % of mid) on top.

Hence the full bid-ask spread estimate is

\boxed{\;\text{Spread} \;=\; 2z \; V \; \sigma_{\sigma}\;\sqrt{\Delta t}\;}

and in vol-points

\text{Spread in vol pts} \;=\; 2z \; \sigma_{\sigma}\;\sqrt{\Delta t}.

⸻

4 .  Practical workflow with just your pricer

step	what you compute	input you need
1	Mid straddle price P	spot S, mid IV \sigma, time-to-exp T (your pricer already gives this)
2	Vega V = P/\sigma	none extra
3	Vol-of-vol \sigma_{\sigma}	pull a time-series of 30-min changes in the same ATM IV and take their annualised st-dev (or use a heuristic: equity-index IV often runs 60 – 80 % p.a.)
4	Choose risk multiple z (e.g. 1) and plug into spread formula	firm’s quoting policy
5	Compare spread with realised $begin:math:text$	\Delta P

If the observed bid–ask is narrower than 2h, liquidity takers have a statistical edge; if it is wider, dealers are over-protected (or inventory/funding is expensive).

⸻

5 .  Does your original idea make sense?
	•	Yes: using P \approx \text{const}\,S\sigma\sqrt{T} is exactly the same as P=V\sigma; it isolates the part of the price driven purely by implied vol.
	•	But remember gamma: if the underlying is very jumpy, spot convexity adds extra uncertainty that widens the economically justified spread.  For most large-cap equity underlyings over 30 minutes, the vega term dominates, so the simplified method above is fine.
	•	You still need some estimate of vol-of-vol; without it you cannot turn price variability into a numeric spread.

⸻

6 .  Quick back-of-the-envelope example

input	value
Spot S	4 500
IV \sigma	22 % p.a.
T	30 days = 0.0822 yr
Mid price P	0.7979\times4500\times0.22\times\sqrt{0.0822}=292
Vega V	P/\sigma = 1327
Vol-of-vol \sigma_{\sigma}	70 % p.a.
\Delta t	0.000305 yr
z	1

\text{Spread}=2\times1\times1327\times0.70\times\sqrt{0.000305}\approx\mathbf{32.5}

So you would expect a $32–33 wide 2-sided market, i.e. $16 each side of mid.
Check this against your screen quotes and against the empirical |ΔP| you measure in the tape; refine z or \sigma_{\sigma} as needed.

⸻

7 .  Caveats & extensions
	•	Vol smiles: use the smile-consistent vega of your pricer, not the flat-vol BS vega, when you are off-ATM.
	•	Funding / inventory costs: dealers often add a fixed bps premium; calibrate by comparing the formula to visible screens.
	•	Event risk windows: around earnings/FOMC the short-term \sigma_{\sigma} spikes; recompute with event-window data.
	•	Microstructure noise: for very tight quoting in liquid indices, quote-step granularity (e.g. $0.05) may be the true lower bound.

⸻

Bottom line:
Using P=V\sigma and a realistic estimate of vol-of-vol gives a clean, data-light way to convert 30-minute price variability into a fair bid–ask spread.  It lines up with standard market-making practice and should serve well as an internal benchmark for whether spreads look rich or cheap.