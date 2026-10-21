
# Netflix Movie Recommendation Engine (SVD)

## Overview
A collaborative filtering recommender that predicts how a user would rate
movies they haven't seen and recommends the highest-rated ones. It uses
Singular Value Decomposition (SVD) to learn hidden user and movie patterns
from a sparse user-movie rating matrix.

## Dataset
- Netflix Prize data (`combined_data_1`): ~24M ratings, 4,499 movies,
  470,758 customers, ratings from 1 to 5
- Movie titles file for mapping IDs to names
- Source: [Kaggle dataset link]

## Approach
1. Parsed the raw file format and mapped each rating to its movie ID
2. Removed users and movies with very few ratings (below the 60th
   percentile: fewer than 908 ratings per movie, fewer than 36 per user),
   leaving ~19.7M ratings
3. Built the model with `surprise.SVD` on a 1M-rating sample
4. Evaluated with 3-fold cross-validation (RMSE)
5. Predicted ratings for all titles for a sample user and ranked the top 15

## Results
| Metric | Value |
|---|---|
| 3-fold CV RMSE | ~0.968 |

## Tools Used
Python, Pandas, NumPy, Scikit-Surprise

## Future Improvements
- Train on the full filtered dataset instead of a sample
- Exclude movies the user has already rated from recommendations
- Tune SVD hyperparameters (factors, epochs, learning rate)
- Compare with KNN-based collaborative filtering
