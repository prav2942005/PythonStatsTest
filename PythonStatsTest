import numpy as np
import pandas as pd
from scipy import stats
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import anova
import datetime


#s1 = np.array([86, 47, 45, 47, 40])
#s2 = np.array([86, 47, 45, 47, 40, 97, 98, 75, 65, 83])
#print(s1.mean())
#print(np.median(s1))
#print(stats.mode(s1))
#print(np.ptp(s1))
#print(np.percentile(s2, 45, interpolation='lower'))
#print(np.percentile(s2, [25,50,75], interpolation='lower'))
#print(stats.iqr(s2, rng=(25,75), interpolation='lower'))
#print(np.var(s2))
#print(np.std(s2))
#print(stats.skew(s2))
#print(stats.kurtosis(s2))
#print(stats.mode([8, 9, 8, 7, 9, 6, 7, 6]))


def teststats1():
    s = np.array([26, 15, 8, 44, 26, 13, 38, 24, 17, 29])
    with open('output.txt', 'w') as f:
        f.write(str(s.mean()) + "\n")
        f.write(str(np.median(s)) + "\n")
        f.write(str(stats.mode(s)) + "\n")
        f.write(str(np.percentile(s, [25, 75], interpolation='lower')) + "\n")
        f.write(str(stats.iqr(s, rng=(25, 75), interpolation='lower')) + "\n")
        f.write(str(stats.skew(s)) + "\n")
        f.write(str(stats.kurtosis(s)) + "\n")


#teststats1()
#np.random.seed(100)
#print(np.random.rand(2, 3))
#print(np.random.choice([11, 22, 33], 2, replace=False))
#x = stats.norm(loc=1.0, scale=2.5)
#print(x.pdf([-1, 0, 1]))
#print(x.cdf([-1, 0, 1]))
#print(x.rvs((2, 3)))


def teststat2():
    np.random.seed(1)
    dist_mean = 32.0
    x = stats.norm.rvs(loc=32.0, scale=4.5, size=100)
    sample_mean = np.mean(x)
    diff_mean = abs(sample_mean - dist_mean)
    with open('output.txt', 'w') as f:
        f.write(str(diff_mean))


#m, s = 0.8, 0.5
#x = stats.norm(m, s)
#sample = x.rvs(100)
#t, p = stats.ttest_1samp(sample, 1.0)
#print(t, p)
#x1 = stats.norm(0.25, 1.0)
#x2 = stats.norm(0.5, 1.0)
#sample1 = x1.rvs(100)
#sample2 = x2.rvs(100)
#t, p = stats.ttest_ind(sample1, sample2)
#print(t, p)

def teststat3():
    s1 = [45, 38, 52, 48, 25, 39, 51, 46, 55, 46]
    s2 = [34, 22, 15, 27, 37, 41, 24, 19, 26, 36]
    t, p = stats.ttest_ind(s1, s2)
    print(t)
    #print("\n")
    print(p)
    s1 = [12, 7, 3, 11, 8, 5, 14, 7, 9, 10]
    s2 = [8, 7, 4, 14, 6, 7, 12, 5, 5, 8]
    t, p = stats.ttest_rel(s1, s2)
    with open('output.txt', 'w') as f:
        f.write(str(t) + "\n")
        f.write(str(p))


#y = np.array([1, 2, 3, 4, 5])
#x1 = np.array([6, 7, 8, 9, 10])
#x2 = np.array([11, 12, 13, 14, 15])
#X = np.vstack([np.ones(5), x1, x2, x1*x2]).T
#print(y)
#print(X)
#data = {'y':y, 'x1':x1, 'x2':x2}
#y, X = patsy.dmatrices('y ~ 1 + x1 + x2 + x1*x2', data)
#print(y)
#print(X)
#bc_cancer_set = sm.datasets.cancer
#bc_cancer = bc_cancer_set.load()
#bc_cancer_data = bc_cancer.data
#print(type(bc_cancer_data))
#icecream_data = sm.datasets.get_rdataset('Icecream', 'Ecdat')
#data1 = icecream_data.data
#print(data1.columns)
#linear_model1 = smf.ols('cons ~ price + temp', data1)
#linear_result1 = linear_model1.fit()
#print(linear_result1.summary())
#linear_model2 = smf.ols('cons ~ income + temp', data1)
#linear_result2 = linear_model2.fit()
#print(linear_result2.summary())
#linear_model3 = smf.ols('cons ~ -1 + income + temp', data1)
#linear_result3 = linear_model3.fit()
#print(linear_result3.score())


def teststat4():
    mtcars_data = sm.datasets.get_rdataset('mtcars').data
    linear_model = smf.ols('np.log(mpg) ~ np.log(wt)', mtcars_data)
    linear_result = linear_model.fit()
    with open('output.txt', 'w') as f:
        f.write(str(linear_result.rsquared))


#df = sm.datasets.get_rdataset('iris').data
#print(df.Species.unique())
#print(df.info())
#df_subset = df[(df.Species == 'versicolor') | (df.Species == 'virginica')].copy()
#print(df_subset.Species.unique())
#df_subset.Species = df_subset.Species.map({'versicolor':1, 'viriginica':0})
#df_subset.rename(columns={"Sepal.Length": "Sepal_Length", "Sepal.Width": "Sepal_Width",	"Petal.Length": "Petal_Length", "Petal.Width": "Petal_Width"}, inplace=True)
#model = smf.logit("Species ~ Petal_Length + Petal_Width", data=df_subset)
#biopsy = sm.datasets.get_rdataset("biopsy","MASS")
#biopsy_data = biopsy.data
#biopsy_data.rename(columns={"class":"Class"},inplace=True)
#biopsy_data.Class = biopsy_data.Class.map({"benign":0,"malignant":1})
#biopsy_data["V1"] = np.divide(biopsy_data["V1"] - biopsy_data["V1"].min(), biopsy_data["V1"].max() - biopsy_data["V1"].min())
#log_mod1 = smf.logit("Class~V1",biopsy_data)
#log_res1 = log_mod1.fit()
#with open('output.txt', 'w') as f:
#    f.write(str(log_res1.prsquared))


#awards_df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")
#print(awards_df.head(3))
#poisson_model = smf.poisson('num_awards ~ math + C(prog)', awards_df)
#poisson_result = poisson_model.fit()
#print(poisson_result.summary())

def teststat5():
    insurance_dataset = sm.datasets.get_rdataset('Insurance','MASS')
    insurance_data = insurance_dataset.data
    poisson_model = smf.poisson('Claims ~ np.log(Holders)', insurance_data)
    poisson_result = poisson_model.fit()
    with open('output.txt', 'w') as f:
        f.write(str(np.sum(poisson_result.resid)))


#icecream = sm.datasets.get_rdataset("Icecream", "Ecdat")
#icecream_data = icecream.data
#model1 = smf.ols('cons ~ temp', icecream_data).fit()
#print(anova.anova_lm(model1))
#model2 = smf.ols('cons ~ income + temp', icecream_data).fit()
#print(anova.anova_lm(model2))
#print(stats.f.sf(31.81, 2, 27))
#print(anova.anova_lm(model1, model2))


def teststat6():
    mtcars_data = sm.datasets.get_rdataset('mtcars').data
    linear_model = smf.ols('np.log(mpg) ~ np.log(wt)', mtcars_data).fit()
    anova_result = anova.anova_lm(linear_model)
    print(anova_result.F['np.log(wt)'])
