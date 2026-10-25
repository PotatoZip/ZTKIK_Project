"""fraud_ae – Project 12 (ZTKIK): fraud detection with a deep autoencoder.

Pipeline phases (mirroring the course abstract):
    1. ``data``      – preprocess the Credit Card Fraud dataset
    2. ``train``     – train the autoencoder on legitimate transactions only
    3. ``threshold`` – pick the reconstruction-error threshold on legitimate validation data
    4. ``evaluate``  – measure recall / FPR on the held-out test set
Baselines (PCA reconstruction, Isolation Forest) live in ``baselines``.
"""

__version__ = "0.1.0"
