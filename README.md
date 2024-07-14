***DESIGN PATTERN AND PROGRAMMING PARADIGM***

    1. Design Pattern - the Python file have a corresponding Jupyter Notebook file, so an interactive jupyter
                        notebook code layout has been adopted
    
    2. Software Paradigm - a Functional paradigm has been adopted
    
    3. Work in progress - the code has been developed merely to show case coding abilities, nor is it designed
                          according to software beased parctices.

***PROJECT DESCRIPTION AND ITS MOTIVATION***
  
    1. We present a model building framework, functions (for ease of unit testing) and algorithms in python for
       various problems for the development of credit risk models - Probability of Default (Application score card)
      
    2. Estimation of Probability of Default is modelled through Statistical methods and Machine learning
    
    3. Mathematical object of interest: The (p+1)-dimensional joint Conditional Probability distribution where the
       vector of independent variables has p-dimension and dependent variable has 1-dimension.

    4. We proceed by modelling the mean of the distribution under the assumption that the observed responses are draw
       from a particular statistical model - in this case the Binomial distribution

    5. We also seek a non-linear function from the vector space of independent variables to a labeled set:(0,1)
       through the greedy algorithm under decision trees.
      
      
***Resources***
 
    1. Resources - https://doi.org/10.1177%2F1536867X1301300407, Little Test of Missing Completely at Random
    by Cheng Li: only up to section 2.2 of the paper has been applied.
      
    2. Resources - http://www.ce.memphis.edu/7012/L17_CategoricalVariableAssociation.pdf, Point Biserial Test
       by University of Memphis

    3.  Resources - Developing Credit Risk Models by Dr. Iain Brown

    4.  Resources - Generalized Linear Models with Examples in R by Peter K Dunn and Gordon K Smith

    5.  Resources - https://www.statlect.com/, Lectures by Marc Taboga  
   
***DATA COLLECTION AND DATA DEFINITIONS***
  
    1. Data definitions and collection methods are detailed in Iain Brown book referenced above.
    
***Libraries and Settings***

    1. Libraries
    2. Settings
  
***DATA DOWNLOAD AND CLEANING***
 
     1. Data Download
     2. Data Cleaning
 
***ANALYSIS OF MISSING VALUES AND IMPUTATION***
 
      1. Missing values count per column
      2. Visualization of missing values patterns

 **Missing values analysis:**

      1. Little Test hypothesis - MCAR test (missing completely at random)
      2. MCAR vs (MNAR, MAR) - Adhoc tests
      3. Imputation via python Simple imputer and KNN

            1.1 Little's Test for MCAR -- Hypothesis testing

                  1.1.1 Resources: Little's Test of Missing Completely at Random by Cheng Li, Northwestern University,
                        Evanston, IL

                  1.1.2 Algorithm presented below

                              1.1.2.1 Inputs
                              1.1.2.2 The Test

             2.1. MCAR adhoc tests vs MNAR, MAR

                   2.1.1 Plots
                   2.1.2 Tests

             3.1 Simple Imputation -- through Python API's

                   3.1.1 KNN Imputation

**concatenate** the imputed dataframes(categorical/float) into one **total dataframe** for further analysis

***EXPLORATORY DATA ANALYSIS***

      1. Hypothesis Tests and Visual Plots:

          1.1  Pearson correlation - numeric variables
          1.2. Chi Square test - categorical variables
          1.3. Point Bisserial test - categorical and numeric variables
          1.4. Correlation and variance inflation factor (VIF)
  
***Point Biserial Test for Binary vs Numerical***

      1. Plot
      2. The Test

***Categorical vs Categorical Chi-square test and plots***

      1. Plot
      2. The Test

***Pearson correlation test for Numerical variables***

      1. Plot
      2. The Test

***Multicollinearity investigation***

      1. VIF Test
  
***DATA CLUSTERING AND DIMENSION REDUCTION***

      1. K_Prototype Clustering
      2. K_Prototype Plots

***TRAIN AND TESTING SAMPLES***
    
      1. One Hot Encoding - Statistical methods and Machine learning
      2. Train and Testing sample split

          2.1 Sample partitioning into train and testing sets

              2.1.1 Defining Independent and Dependent variables - Statistics
              2.1.2 Sample imbalance investigation
              2.1.3 Training and Testing samples

***GLM FIT-BINOMIAL***

      1. Fit a binomial distribution over data 
  
***MODEL ASSESSMENT***
  
      1. Perfomance measurement

            1.1 ROC Curve Analytics and Optimal threshold
            1.2 Prediction Function @ maximal threshold
            1.3 Confusion Matrix Plot

      2. Diagonostics Tests

            2.1 Hypothesis Tests and Visual Plots:

                    2.1.1 Quantile Residuals - Residuals for Discrete GLMs
                    2.2.2 Breush Pagan Test - Heteroskedasticity of Variance
                    2.3.3 Normal Residuals Test
                    2.4.4 Durbin Watson Test - Test for Errors Serial Correlation
                    2.5.5 Leverage Studentized Quantile Residuals
                    2.6.6 Partial Residuals Plots
                    2.7.7 Cooks Distance Quantile Residuals

***DIAGNOSTICS REMEDIES***

        1. Conduct Hypothesis tests to drop insignificant variables:

              1.1 Wald test
              1.2. Score test
              1.3. Likelihood ratio test

        2. Conduct Chi square test for model fit using Deviance residuals

        3. Investigate Diagnostics checks and respond accordingly per each test by adding/dropping required varaibles

***FINAL STATISTICAL MODEL - after diagnostics remedies***

        1. Refit the GLM after incoporating above changes

***MODEL ALTERNATIVES - Decision Tree***
  
        1. Base Tree

                1.1 Fit a base tree
                1.2 Base tree prediction
                1.3 Base tree Confusion matrix
                1.4 Base tree plot

       2. Pruned tree by Cross Validation

                2.1 Extracting alphas for pruning
                2.2 Cross validation for best alpha
                2.3 Extraction of ideal alpha

       3. Final Tree fitting
      
                3.1 Ideal alpha value for pruning the tree
                3.2 Pruned tree fitting

        4. Prediction and perfomance analytics

                 4.1 Predict
                 4.2 Confusion matrix plot
                 4.3 Plot final tree
  
***ANALYSIS OF REJECT INFERENCE AND FINAL COMPLETE MODEL - through Decision Tree***
   
       1. Data download
       2. Data cleaning

       3. Missing values analysis and imputation
       4. Data Clustering with K-Prototypes

       5. Prediction with Pruned Decision Tree to determine what would have been Bad/Good customers

             5.1 Create Testing Dataframe (the whole dataframe)
             5.2 Run prediction
             5.3 Final Dataframes - Float and Categorical

       6. Fit a Decision Tree

               6.1 Creating independent and dependent variables
               6.2 Sample partition into Train and Test sets
               6.3 Find ideal alpha through cross validation
               6.4 Fitting a pruned tree

       7. Perfomance and Goodness of fit of the model

             7.1 Prediction
             7.2 Confusion matrix plot
             7.3 Decision tree plot
   
***MODEL DEPLOYMENT***
  
        1. Save with pickle

        2. Options:

                2.1. Cloud deployment
                2.2. Build a GUI for the model
