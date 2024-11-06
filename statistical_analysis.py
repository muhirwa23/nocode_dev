# statistical_analysis.py

from scipy import stats
import statsmodels.api as sm

def t_test(df, col1, col2):
    """Performs T-Test between two columns."""
    return stats.ttest_ind(df[col1], df[col2])

def chi_square_test(df, col1, col2):
    """Performs Chi-Square Test between two columns."""
    contingency_table = pd.crosstab(df[col1], df[col2])
    return stats.chi2_contingency(contingency_table)

def anova(df, *columns):
    """Performs ANOVA test between multiple columns."""
    return stats.f_oneway(*(df[col] for col in columns if col in df.columns))

def linear_regression(X, y):
    """Fits a linear regression model."""
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()
