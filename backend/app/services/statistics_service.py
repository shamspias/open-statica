import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from typing import Dict, Any, List


class StatisticsService:
    """Simplified statistics service for essential analyses"""

    async def descriptive_statistics(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Calculate descriptive statistics"""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                data = df[col].dropna()
                results[col] = {
                    'count': int(data.count()),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'mode': float(data.mode()[0]) if len(data.mode()) > 0 else None,
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q1': float(data.quantile(0.25)),
                    'q3': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis())
                }
            else:
                # Categorical variable
                value_counts = df[col].value_counts()
                results[col] = {
                    'count': int(df[col].count()),
                    'unique': int(df[col].nunique()),
                    'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'frequency': value_counts.head(10).to_dict()
                }

        return results

    async def correlation_analysis(self, df: pd.DataFrame, columns: List[str], method: str = 'pearson') -> Dict[
        str, Any]:
        """Calculate correlations"""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[columns].corr(method=method)

        # Find significant correlations
        significant = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    significant.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        return {
            'matrix': corr_matrix.to_dict(),
            'significant': significant
        }

    async def t_test(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform t-test"""
        test_type = request.get('type', 'independent')

        if test_type == 'independent':
            group_col = request.get('group_column')
            value_col = request.get('value_column')

            groups = df[group_col].unique()
            if len(groups) != 2:
                return {'error': 'Need exactly 2 groups for t-test'}

            group1 = df[df[group_col] == groups[0]][value_col].dropna()
            group2 = df[df[group_col] == groups[1]][value_col].dropna()

            t_stat, p_value = stats.ttest_ind(group1, group2)

            return {
                'test_type': 'Independent t-test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'group1_mean': float(group1.mean()),
                'group2_mean': float(group2.mean()),
                'significant': p_value < 0.05
            }

        elif test_type == 'paired':
            var1 = request.get('variable1')
            var2 = request.get('variable2')

            t_stat, p_value = stats.ttest_rel(df[var1].dropna(), df[var2].dropna())

            return {
                'test_type': 'Paired t-test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        return {'error': 'Invalid test type'}

    async def anova(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one-way ANOVA"""
        group_col = request.get('group_column')
        value_col = request.get('value_column')

        groups = []
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)

        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for ANOVA'}

        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'n_groups': len(groups),
            'significant': p_value < 0.05
        }

    async def chi_square_test(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform chi-square test"""
        var1 = request.get('variable1')
        var2 = request.get('variable2')

        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'contingency_table': contingency_table.to_dict()
        }

    async def regression(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regression analysis"""
        regression_type = request.get('type', 'linear')
        predictors = request.get('predictors', [])
        target = request.get('target')

        # Prepare data
        X = df[predictors].dropna()
        y = df[target].dropna()

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if regression_type == 'linear':
            # Use statsmodels for detailed output
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()

            return {
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'r_squared': float(model.rsquared),
                'adjusted_r_squared': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue)
            }

        elif regression_type == 'logistic':
            # Simple logistic regression
            model = LogisticRegression(max_iter=1000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            accuracy = accuracy_score(y_test, model.predict(X_test))

            return {
                'accuracy': float(accuracy),
                'coefficients': dict(zip(predictors, model.coef_[0])),
                'intercept': float(model.intercept_[0])
            }

        return {'error': 'Invalid regression type'}

    async def clustering(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform K-means clustering"""
        features = request.get('features', [])
        n_clusters = request.get('n_clusters', 3)

        # Prepare data
        X = df[features].dropna()

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        return {
            'n_clusters': n_clusters,
            'cluster_labels': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_)
        }

    async def classification(self, df: pd.DataFrame, request: Dict[str, Any]) -> Dict[str, Any]:
        """Train simple classification model"""
        features = request.get('features', [])
        target = request.get('target')

        # Prepare data
        X = df[features].dropna()
        y = df[target].dropna()

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))

        return {
            'accuracy': float(accuracy),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'feature_importance': dict(zip(features, np.abs(model.coef_[0])))
        }
