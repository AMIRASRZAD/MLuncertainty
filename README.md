# MLuncertainty
Explored aleatoric and epistemic uncertainty and compared the correlation with other uncertainty metrics 
We have focused on tabular binary classification problems and first step is to detect the suspicious points. Initially we employed trust score to recognize the unreliable decisions of the black box ML. However, we observe that trust score for some instances that are located in the suspicious area near to the decision boundary is unreasonably high and also other data points that are located far from decision boundary getting similar trust as points near to decision boundary. To better understand the reason, we need to look precisely in trust score formula. Let’s move on to the steps; first step is to filtering the training data based on their density and selecting high density points:
Suppose D_N={[x_i1,…,x_im ],y_i }_(i=1)^N={x_i,y_i }_(i=1)^N be the set of training and validation data for building the black box model h. Without loss of generality, assume that y_i={0,1},∀i.
assume that we have f0 and f1 density function for each label. 
The estimate of high-density points for x instance can be obtained by following equations (Jiang et al. 2018):
(H_α ) ̂{f} ∶= {x∈X: r_k (x)≤ ϵ}  (1)
Where:
r_k (x)≔inf⁡{r>0∶|B(x,r)∩X|≥k}  (2)
is called k-NN radius of instance x. ( B(x,r)  is the ball around x with r radius). And
 ϵ∶=inf⁡{r>0:|{x∈X: r_k (x)>r}|≤ α.n}  (3)
is the minimum radius obtained by predefined density threshold α.
Then trust score for instance x can be defined as: 
ts=d(x,(H_α ) ̂{f_(h ̂(x)) })/d(x,(H_α ) ̂{f_(h(x)) })  (4)
Where the  d is a distance function and h(x) is the predicted class for the x by h and h ̂(x) is the opposite class.
![image](https://github.com/AMIRASRZAD/MLuncertainty/assets/30290184/5798e295-e231-4772-80a8-f24aab37a96f)
