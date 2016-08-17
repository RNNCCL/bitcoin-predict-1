
import statsmodels.api as sm

def anova_table():
    print sm.stats.anova_lm(
        sm.formula.ols("acc ~ C(rep_name) * C(clf_name)", data=df_results).fit()
    ).to_latex()
    print sm.stats.anova_lm(
        sm.formula.ols("roc_auc ~ C(rep_name) * C(clf_name)", data=df_results).fit()
    ).to_latex()
