## Project 12: Anomaly and Fraud Detection in Bank Transactions Using an Autoencoder[cite: 4]

*English Name: Project 12: Financial Fraud Detection Using Deep Autoencoders*[cite: 4]

**Glossary of Abbreviations: AE (Autoencoder), Fraud Detection, MSE (Mean Squared Error).**[cite: 4]

| Category | Details |
| :--- | :--- |
| **Description and Research Goal:** | Training a deep Autoencoder in PyTorch exclusively on legitimate bank transactions. Fraud detection is performed based on a high reconstruction error (MSE).[cite: 4] |
| **Formal Threat Model:** | Financial Fraud / Anomaly Attacker. The adversary executes unauthorized transactions breaking standard limits.[cite: 4] |
| **Scenario / Context:** | Automated anti-fraud systems in electronic banking and payment cards.[cite: 4] |
| **Implementation Phases (Phases):** | 1. Preprocessing the Credit Card Fraud database; 2. Training the Autoencoder on legitimate transactions; 3. Determining the MSE error threshold; 4. Evaluating sensitivity to fraudulent transactions.[cite: 4] |
| **Attack Attempt / Criterion:** | Detecting over 85% of fraudulent transactions while maintaining a false alarm rate < 1%.[cite: 4] |
| **Database & Link:** | Credit Card Fraud Detection Dataset – https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud[cite: 4] |

### Scientific Literature (IEEE/APA):[cite: 4]
* An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection using reconstruction probability. Special Lecture on IE, 2(1), 1-18.[cite: 4]
* ULB Machine Learning Group (2024). Credit Card Fraud Detection Dataset. Kaggle.[cite: 4]
* PyTorch Documentation (2024). Building Autoencoders with PyTorch. https://pytorch.org[cite: 4]