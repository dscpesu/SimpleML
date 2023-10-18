The Issue asks us to predict the "Score_pos_neg_diff" column using sentiment analysis of amazon reviews with the help of RandomForestClassifier.

Firstly, I have built the sentiment analysis model of Amazon reviews with the help of the RandomForestClassifier model. This works pretty well in classifying the Amazon reviews either as Positive, Negative or Neutral with an accuracy of 91%.

Next, to predict the "Score_pos_neg_diff" column. This column is slightly tricky to predict, and below are some of my analysis:
- The "Score_pos_neg_diff" column essentially depends on the Helpful_yes and Helpful_no columns. It is the difference between the yes and no columns.
- Our target column is the "Score_pos_neg_diff" column. Our matrix of features, contains all the columns other than the target column.
- When this is the case, we really don’t have anything to predict, or use machine learning Algorithms, as the value of the target attribute can easily be obtained using a simple subtraction operation.
- Next, the dataset contains 4913 rows, out of which 4501 rows do not have a helpful_yes vote and 4673 rows do not have a helpful_no vote. This is a large percentage of the dataset, and as our target variable is predicted using these two columns, we are expected to receive inaccurate results.
- Next, when we plot a Correlation plot between the target and other variables, we can see that the column sentiment_encoded is not at all correlated with the score_pos_neg_diff column and is neither correlated with helpful_yes nor helpful_no column. This tells us that, for predicting the score_pos_neg_diff column, we do not really need the sentiment analysis part.
- Thus, I have Implemented data preprocessing, embeddings, and RandomForest classifier, which is able to predict the sentiment of a review text into its appropriate sentiment with an accuracy of about 91% on the test set.
- Thus, the issue of classifying sentiments using RandomForest classifier has been completed successfully.

The issue of predicting the score_pos_neg_diff column needs to be reconsidered by the owner, because from my end, with the existing dataset and above reasonings, it is indeed ambiguous and unlikely to be able to predict this target attribute.
