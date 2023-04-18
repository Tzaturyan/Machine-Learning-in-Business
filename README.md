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
