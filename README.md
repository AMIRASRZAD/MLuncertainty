In these experiments, we explored the relationship between the trust score and two distinct types of uncertainty in machine learning predictions: epistemic and aleatoric. Aleatoric uncertainty arises from the random nature of classification and is considered irreducible. This form of uncertainty intensifies in classification decisions within regions where data points from different classes overlap. In contrast, epistemic uncertainty, which is reducible, stems from a lack of knowledge and training points in the vicinity of the data points under examination (Hüllermeier and Waegeman 2021).
In our experiments, we computed both types of uncertainty by employing deep ensemble models across various classification datasets. Additionally, we incorporated the Brier score, another metric capturing uncertainty. We examined their correlations to understand their relative behaviors. Deep ensembles were chosen as our model of choice due to their capability to capture both epistemic and aleatoric uncertainties (Lakshminarayanan et al. 2017).

 
      
![image](https://github.com/AMIRASRZAD/MLuncertainty/assets/30290184/902ca043-ee93-40a4-bc0e-8c00575d383e)
