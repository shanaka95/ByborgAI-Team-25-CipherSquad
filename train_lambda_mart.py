import os
import csv
import logging
import pickle
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import ndcg_score
    ADVANCED_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Some advanced dependencies are not available. Please install them with:")
    logger.warning("pip install pandas lightgbm scikit-learn")
    ADVANCED_IMPORTS_AVAILABLE = False

class LambdaMARTTrainer:
    """
    Trains a LambdaMART model for learning to rank using LightGBM.
    """
    
    def __init__(
        self,
        input_file: str = "preprocessed_user_video_data.csv",
        output_dir: str = "models",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the LambdaMART trainer.
        
        Args:
            input_file: Path to the preprocessed data CSV file
            output_dir: Directory to save the trained model
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_file)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        self.test_size = test_size
        self.random_state = random_state
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.feature_names = None
        
    def load_data(self):
        """Load and preprocess the data for training"""
        if not ADVANCED_IMPORTS_AVAILABLE:
            logger.error("Required packages are not available. Cannot load data.")
            return None, None, None, None
        
        try:
            # Load data
            logger.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(df)} rows of data")
            
            # Filter out rows with missing values
            df = df.dropna()
            logger.info(f"After dropping missing values: {len(df)} rows")
            
            # Convert category_match to numeric (Y=1, N=0)
            df['category_match_numeric'] = df['category_match'].apply(lambda x: 1 if x == 'Y' else 0)
            
            # Encode user IDs for grouping purposes only (not as features)
            user_encoder = LabelEncoder()
            df['user_encoded'] = user_encoder.fit_transform(df['user'])
            
            # Group data by user and video1 (query)
            grouped_data = []
            for (user, video1), group in df.groupby(['user', 'video1']):
                # Only include groups with at least one relevant item
                if group['relevance_label'].max() > 0:
                    # Only include groups with at least 2 items for proper ranking
                    if len(group) >= 2:
                        grouped_data.append(group)
            
            # Combine grouped data
            if grouped_data:
                df_grouped = pd.concat(grouped_data)
                logger.info(f"After grouping: {len(df_grouped)} rows")
            else:
                logger.warning("No valid groups found after filtering")
                return None, None, None, None, None, None
            
            # Define features and target - only use actual content features, not IDs
            features = [
                'description_similarity', 
                'category_match_numeric'
            ]
            
            logger.info(f"Using features: {features}")
            
            X = df_grouped[features]
            y = df_grouped['relevance_label']
            
            # Create group identifiers for ranking
            group_keys = df_grouped[['user_encoded', 'video1']].drop_duplicates()
            group_keys['group_id'] = np.arange(len(group_keys))
            
            # Merge group IDs back to the main dataframe
            df_with_groups = pd.merge(
                df_grouped, 
                group_keys, 
                on=['user_encoded', 'video1'], 
                how='left'
            )
            
            # Extract groups
            groups = df_with_groups['group_id']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X, y, groups, test_size=self.test_size, random_state=self.random_state, 
                stratify=None  # Don't stratify by target
            )
            
            # Count items per group in test set
            test_group_counts = groups_test.value_counts()
            logger.info(f"Test set has {len(test_group_counts)} groups")
            logger.info(f"Min items per group in test set: {test_group_counts.min()}")
            logger.info(f"Max items per group in test set: {test_group_counts.max()}")
            logger.info(f"Avg items per group in test set: {test_group_counts.mean():.2f}")
            
            logger.info(f"Training set: {len(X_train)} rows")
            logger.info(f"Testing set: {len(X_test)} rows")
            
            # Save encoders for later use
            self._save_encoders(user_encoder)
            
            # Save feature names
            self.feature_names = features
            
            # Convert to numpy arrays for easier handling in evaluation
            X_train_np = X_train.values
            X_test_np = X_test.values
            y_train_np = y_train.values
            y_test_np = y_test.values
            groups_train_np = groups_train.values
            groups_test_np = groups_test.values
            
            return X_train_np, X_test_np, y_train_np, y_test_np, groups_train_np, groups_test_np
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None, None, None, None, None
    
    def _save_encoders(self, user_encoder):
        """Save label encoders for later use"""
        encoders = {
            'user_encoder': user_encoder
        }
        
        encoders_path = os.path.join(self.output_dir, "label_encoders.pkl")
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        logger.info(f"Saved label encoders to {encoders_path}")
    
    def train_model(self, X_train, y_train, groups_train, X_test=None, y_test=None, groups_test=None):
        """Train the LambdaMART model using LightGBM"""
        if not ADVANCED_IMPORTS_AVAILABLE:
            logger.error("Required packages are not available. Cannot train model.")
            return None
        
        try:
            # Create group counts for LightGBM
            unique_groups_train = np.unique(groups_train)
            train_group_counts = np.array([np.sum(groups_train == i) for i in unique_groups_train])
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train, 
                label=y_train,
                group=train_group_counts,
                feature_name=self.feature_names
            )
            
            if X_test is not None and y_test is not None and groups_test is not None:
                # Create group counts for test data
                unique_groups_test = np.unique(groups_test)
                test_group_counts = np.array([np.sum(groups_test == i) for i in unique_groups_test])
                
                test_data = lgb.Dataset(
                    X_test,
                    label=y_test,
                    group=test_group_counts,
                    feature_name=self.feature_names,
                    reference=train_data
                )
            else:
                test_data = None
            
            # Set parameters for LambdaMART
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5, 10],
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_data_in_leaf': 20,
                'max_depth': -1,
                'verbose': -1
            }
            
            # Set callbacks for early stopping
            callbacks = []
            if test_data:
                callbacks.append(lgb.early_stopping(stopping_rounds=50))
                callbacks.append(lgb.log_evaluation(period=50))
            
            # Train model
            logger.info("Training LambdaMART model...")
            if test_data:
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, test_data],
                    valid_names=['train', 'test'],
                    callbacks=callbacks
                )
            else:
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    callbacks=[lgb.log_evaluation(period=50)]
                )
            
            # Print feature importance
            if self.model is not None:
                importance = self.model.feature_importance(importance_type='gain')
                feature_importance = sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)
                
                logger.info("Feature importance (gain):")
                for feature, importance in feature_importance:
                    logger.info(f"  {feature}: {importance}")
            
            logger.info("Model training completed")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def evaluate_model(self, X_test, y_test, groups_test):
        """Evaluate the trained model on test data"""
        if not ADVANCED_IMPORTS_AVAILABLE or self.model is None:
            logger.error("Required packages are not available or model is not trained. Cannot evaluate model.")
            return
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate NDCG for each group
            ndcg_scores = []
            unique_groups = np.unique(groups_test)
            
            # Count groups with enough items for each k
            valid_groups_count = {1: 0, 3: 0, 5: 0, 10: 0}
            
            for group_id in unique_groups:
                # Create mask for this group
                group_indices = np.where(groups_test == group_id)[0]
                group_size = len(group_indices)
                
                # Skip groups with only one item
                if group_size < 2:
                    continue
                
                # Get true and predicted relevance for this group
                y_true_group = y_test[group_indices].reshape(1, -1)
                y_pred_group = y_pred[group_indices].reshape(1, -1)
                
                # Calculate NDCG@k for different k values
                for k in [1, 3, 5, 10]:
                    # Only calculate NDCG@k if we have enough items
                    if group_size >= k:
                        try:
                            score = ndcg_score(y_true_group, y_pred_group, k=k)
                            ndcg_scores.append((k, score))
                            valid_groups_count[k] += 1
                        except Exception as e:
                            logger.debug(f"Error calculating NDCG@{k} for group {group_id}: {str(e)}")
            
            # Calculate average NDCG for each k
            ndcg_by_k = defaultdict(list)
            for k, score in ndcg_scores:
                ndcg_by_k[k].append(score)
            
            # Print results
            logger.info("Evaluation results:")
            for k, scores in ndcg_by_k.items():
                avg_score = np.mean(scores)
                logger.info(f"NDCG@{k}: {avg_score:.4f} (calculated on {valid_groups_count[k]} groups)")
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save_model(self, filename: str = "lambdamart_model.txt"):
        """Save the trained model"""
        if not ADVANCED_IMPORTS_AVAILABLE or self.model is None:
            logger.error("Required packages are not available or model is not trained. Cannot save model.")
            return
        
        try:
            # Save model
            model_path = os.path.join(self.output_dir, filename)
            self.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save feature names
            feature_path = os.path.join(self.output_dir, "feature_names.pkl")
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info(f"Feature names saved to {feature_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None
    
    def train_and_evaluate(self):
        """Complete training pipeline"""
        if not ADVANCED_IMPORTS_AVAILABLE:
            logger.error("Required packages are not available. Cannot train model.")
            logger.info("Please install required packages with: pip install pandas lightgbm scikit-learn")
            return
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, groups_train, groups_test = self.load_data()
        
        if X_train is None:
            logger.error("Failed to load data. Cannot train model.")
            return
        
        # Train model
        self.train_model(X_train, y_train, groups_train, X_test, y_test, groups_test)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test, groups_test)
        
        # Save model
        self.save_model()
        
        logger.info("Training and evaluation completed")

def main():
    """Main function to train the LambdaMART model"""
    if not ADVANCED_IMPORTS_AVAILABLE:
        logger.error("Required packages are not available. Cannot train model.")
        logger.info("Please install required packages with: pip install pandas lightgbm scikit-learn")
        return
    
    # Initialize trainer
    trainer = LambdaMARTTrainer(
        input_file="data/preprocessed_user_video_data.csv",
        output_dir="models",
        test_size=0.2,
        random_state=42
    )
    
    # Train and evaluate model
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main() 