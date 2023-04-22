# Fairness in student course completion based on student data

## Introduction

With the increasing adoption of Massive Online Open Courses (MOOC), newer educational systems are being built, which can gauge the needs of students to suggest them courses appropriate for them. One of the key factors these systems could consider is the prediction whether if the student given the course would pass or fail the course. Apart from considering the academic factors, these systems could also take into account the personal factors like age, gender, region and disability in their prediction decision, which poses a risk of being unfair while using these attributes. There is a great scope in building fair educational systems, which can be used to provide courses to all in a fair manner.

In this tutorial, we show how Seldonian algorithm can be used in this context to build a fair online education system which is fair across various student demographics. We use the [OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) here, which contains information about 32593 students and their demographic data, used in predicting whether a student is likely to pass the courses offered by Open University. Open University is a public British University that also has the highest number of undergraduate students in the UK. The data presented here is sourced from the Open University's Online Learning platform.

## Dataset Preparation

The following outline the dataset preparation process of our pipeline. 

- Firstly, we dropped the columns like student ID which have no importance in the predicition pipeline. 
- Secondly, we manipulated the columns like highest education where we grouped divisions like A level or equivalent, post grads, and HE qualification to be a boolean 1 whereas lower than A level and no formal quals to be 0
- We also converted columns like distinction to binaries. 
- The next step is to convert the categorical variables into numerical values. This is done using the LabelEncoder function of the scikit-learn library. The LabelEncoder function assigns a numerical value to each unique categorical value in the column.

- After converting the categorical variables, the next step is to standardize the numerical variables. This is done using the StandardScaler function of the scikit-learn library. The StandardScaler function standardizes the numerical variables by subtracting the mean and dividing by the standard deviation.

Once the preprocessing steps are complete, we save the dataframe and the meta data which is later used in training and experimentation. 

