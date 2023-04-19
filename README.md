# Machine-Learning-in-Business
# 1. Data-driven approach based on the example of the application routing task in helpdesk
# 2. User profiling. Segmentation: unsupervised learning (clustering, LDA/ARTM), supervised (multi/binary classification)
1) Independently figure out what tfidf is (documentation https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html and yet - https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction )
2) Modify the code of the get_user_embedding function so that the median is not considered (as in the np.mean example), but the median. Apply such a transformation to the data, train an outflow prediction model and calculate quality metrics and save them: roc auc, precision/recall/f_score (for the last 3 - select the optimal threshold using precision_recall_curve, as was done in the lesson) Repeat step 2, but using no longer the median, but the max
3) (optional, if you really want to) Using the knowledge gained from paragraph 1, repeat paragraph 2, but already weighing the news by tfidf (hint: you need to get weights coefficients for each document. Not all documents are equally informative and carry some kind of positive signal). Hint 2 - it is the idf that is needed, as a weight.
4) Generate a single table at the output comparing the quality of 3 different methods for obtaining user embeddings: mean, median, max, idf_mean by the metrics roc_auc, precision, recall, f_score
5) Make independent conclusions and assumptions about why this or that method turned out to be more effective than the others
# 3. The relationship between business indicators and DS metrics
1) train several different models on a set of CVD data (train_case2.csv): logreg, boosting, forest, etc. - 2-3 options for your choice
2) when training models, it is mandatory to use cross-qualification
3) display a comparison of the obtained models by the main classification metrics: pr/rec/auc/f_score (it is possible in the form of a table, where the rows are models and the columns are metrics)
4) draw conclusions about which model coped with the task better than others
5) (optional question) which metric (precision_recall_curve or roc_auc_curve) is more suitable in case of a strong imbalance of classes? (when there are many more objects of one of the classes than the other).
# 4. Uplift-моделирование
Download the data set of marketing campaigns from here https://www.kaggle.com/davinwijaya/customer-retention
there, the conversion field is the target variable, and the offer is the communication. Rename the fields (conversion -> target, offer -> treatment) and bring the treatment field to a binary form (1 or 0, i.e. there was some offer or not) - the value No Offer means no communication, and all the others - the presence.
to split the data set into non-training and test samples;
discretion (freedom of choice of methods is allowed)
to carry out uplift modeling in 3 ways: one model with a communication feature (S learner), a model with a target transformation (transformation of classes of clause 2.1) and an option with two independent models
at the end to display a single comparison table of metrics uplift @10%, uplift@20% of these 3 models
build an UpliftTreeClassifier model and try to describe the resulting tree in words;
(optional) for the S learner model (a model with an additional sign of communication), build a dependence of the target (conversion - conversion field) on the uplift value: 1) make a forecast and get an uplift for the test sample 2) sort the test sample by uplift in descending order 3) split into deciles (pandas qcut to help you) 4) calculate the average conversion for each decile
(optional) build an UpliftRandomForestClassifier model and try to describe the resulting tree in words
# 5. The outflow problem: formulation options, possible solutions
For our pipeline (Case1), experiment with different models: 1 - boosting, 2 - logistic regression (do not forget to add standardization to cont_transformer here - normalization of real features)
Select the best model by metrics (by the way, which in your opinion is the most suitable DS metric here)
deferred sample), make an assessment of economic efficiency with the same introductory data as in question 2 ($ 1 for attraction, $ 2 for each correctly classified (True Positive) retained). (hint) you need to calculate FP/TP/FN/TN for the selected optimal probability threshold and calculate revenue and expenses.
(optional) To carry out the selection of hyperparameters of the best model based on the results of 2-3
(optional) To evaluate the economic efficiency once again
