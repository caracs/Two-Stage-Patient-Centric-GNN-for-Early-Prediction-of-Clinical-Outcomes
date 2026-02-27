# Two-Stage-Patient-Centric-GNN-for-Early-Prediction-of-Clinical-Outcomes

# Early Identification of Clinical Deterioration via Graph-Based Deep Learning

Early identification of clinical deterioration enables timely transfer to the Intensive Care Unit (ICU) and can significantly improve patient outcomes.  

In this work, I propose a two-stage deep learning framework that transforms Electronic Health Records (EHR) from the MIMIC-IV database into a patient-level graph to jointly predict multiple ICU-related outcomes.

## Stage 1 – Clinical State Estimation

A Transformer-based **Clinical State Estimator** processes each hospital admission and outputs:

- The probability of subsequent ICU transfer  
- The expected time to transfer  

The resulting latent representations and predicted risks are aggregated across admissions to construct a heterogeneous patient graph, enriched with ICU vital sign statistics for patients who eventually reach the ICU.

Graph edges encode:

- Feature-space similarity  
- Intra-patient temporal relationships  

## Stage 2 – Graph-Based Outcome Prediction

Patient-level outcome prediction is performed using a hybrid:

- GCN  
- GATv2  
- GraphSAGE  

model with **Jumping Knowledge aggregation**.

The model simultaneously forecasts:

- In-hospital mortality (classification)
- ICU length of stay (LOS) (regression)

## Results

On the MIMIC-IV dataset, the approach achieves strong performance:

- **Mortality prediction**:  
  - AUROC: 0.932  
  - Accuracy: 0.959  

- **LOS regression**:  
  - RMSE: 59 h  
  - MAE: 38 h  

An ablation study demonstrates that:

- The multi-edge graph design  
- The multitask learning objective  

both contribute substantially to performance improvements.

## Conclusion

The method:

- Relies solely on routinely collected hospital data  
- Is fully end-to-end differentiable  
- Supports real-time execution on a single GPU  

These properties highlight its potential for prospective deployment as a graph-aware clinical decision support tool.
