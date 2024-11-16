import warnings
import pandas as pd  # for data manipulation and analysis, CSV file I/O
import numpy as np  # For numerical operations and mathematical functions
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical graphics
from sklearn.model_selection import train_test_split  # For data splitting (Training & Testing) in machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For feature standardization & Normalization
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
from fpdf import FPDF

loan_data = pd.read_csv(r"C:\Users\Lenovo\Downloads\train_u6lujuX_CVtuZ9i (1).csv")
print(loan_data.head())
print("The shape =", loan_data.shape)

# Dataset dimensions and statistics
num_rows, num_cols = loan_data.shape
num_features = num_cols - 1
num_data = num_rows * num_cols

# Print the information about the dataset
print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_cols}")
print(f"Number of Features: {num_features}")
print(f"Number of All Data: {num_data}")
print(loan_data.info())
print(loan_data.describe().T.round(2))
print(loan_data.describe(include=object))
##Graphical Analysis
GenderAnalysis = loan_data.Gender.value_counts(dropna=False)
print(GenderAnalysis)

# Bar Charts Analysis "For Gender feature"
sns.countplot(x="Gender", data=loan_data, legend=False)
plt.show()

# "dropna" as False to count NaN values
MarriedAnalysis = loan_data.Married.value_counts(dropna=False)
print(MarriedAnalysis)

# Create a pie chart for the Married feature
plt.figure(figsize=(10, 5))  # figure in inches

# Create labels based on the index of MarriedAnalysis
labels = MarriedAnalysis.index.astype(str)  # Convert index to string to handle NaN

plt.pie(MarriedAnalysis,
        labels=labels,
        startangle=216,
        autopct='%1.1f%%'
       )

plt.axis('equal')  # Used to set the aspect ratio of the plot to be equal.
plt.title('Marital Status Distribution')
plt.show()
DependentsAnalysis = loan_data.Dependents.value_counts(dropna=False)
print(DependentsAnalysis)

# Bar Charts Analysis "For Dependents feature"
sns.countplot(x="Dependents", data=loan_data,legend=False)
plt.show()


##data relationship analysis
pd.crosstab(loan_data.Credit_History, loan_data.Loan_Status).plot(kind="bar", figsize=(8, 4))

# Add a title to the plot
plt.title('Credit History VS Loan Status')

# Label the x-axis
plt.xlabel('Credit History')

# Label the y-axis
plt.ylabel('Count')

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=0)

# Display the plot
plt.show()

# Create a cross-tabulation of 'Property Area' and 'Loan Status' variables
pd.crosstab(loan_data.Property_Area, loan_data.Loan_Status).plot(kind="bar", figsize=(8, 4))

# Add a title to the plot
plt.title('Property Area VS Loan Status')

# Label the x-axis
plt.xlabel('Property Area')

# Label the y-axis
plt.ylabel('Count')

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=0)

# Display the plot
plt.show()

# Calculate the average income
average_income = loan_data['ApplicantIncome'].mean()
print(f"The Average Income: {average_income:.2f} ")

# Count incomes higher and lower than average
above_average_count = (loan_data['ApplicantIncome'] > average_income).sum()
below_average_count = (loan_data['ApplicantIncome'] <= average_income).sum()

# Calculate ratio and print the results
ratio = above_average_count / below_average_count
print(f"The ratio of people with income above average to below average: {ratio*100:.2f} ")
print(f"Number of people income above the average: {above_average_count}")
print(f"Number of people income below the average: {below_average_count}")
plt.figure(figsize=(8, 4))
sns.barplot(x=['Above Average', 'Below Average'], y=[above_average_count, below_average_count], legend=False)
plt.title('Ratio of People with Income Above Average to Below Average')
plt.ylabel('Count')
plt.show()
Credit_HistoryAnalysis = loan_data.Credit_History.value_counts(dropna=False)
print(Credit_HistoryAnalysis)

# Bar Charts Analysis "For Credit History feature"
sns.countplot(x="Credit_History", data=loan_data, legend=False)
plt.show()

Property_AreaAnalysis = loan_data.Property_Area.value_counts(dropna=False)
print(Property_AreaAnalysis)

# Bar Charts Analysis "For Property Area feature"
sns.countplot(x="Property_Area", data=loan_data, legend=False)
plt.show()

def plot_distribution(column, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=loan_data, x=column, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Plot distribution for each numerical column
plot_distribution("ApplicantIncome", "Applicant Income Distribution")
plot_distribution("CoapplicantIncome", "Coapplicant Income Distribution")
plot_distribution("LoanAmount", "Loan Amount Distribution")
plot_distribution("Loan_Amount_Term", "Loan Amount Term Distribution")
# Correlation matrix using heatmap
# Calculates the correlation coefficients between all pairs of numerical variables in the dataset
correlation_matrix = loan_data.corr(numeric_only=True)

# Create a heatmap figure with specified size (in inches)
plt.figure(figsize=(15, 7.5))

# Generate a heatmap to visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True) # annot: write the data value in each cell

# Add a title to the plot
plt.title('Correlation Matrix')

# Display the heatmap
plt.show()

# Create a cross-tabulation of 'Gender' and 'Married' variables
pd.crosstab(loan_data.Gender, loan_data.Married).plot(kind="bar", figsize=(8, 4))

# Add a title to the plot
plt.title('Gender VS Married')

# Label the x-axis
plt.xlabel('Gender')

# Label the y-axis
plt.ylabel('Count')

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=0)

# Display the plot
plt.show()

# Create a cross-tabulation of 'Gender' and 'Loan Status' variables
pd.crosstab(loan_data.Education, loan_data.Loan_Status).plot(kind="bar", figsize=(8, 5))

# Add a title to the plot
plt.title('Education Status VS Loan Status')

# Label the x-axis
plt.xlabel('Education')

# Label the y-axis
plt.ylabel('Count')

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=0)

# Display the plot
plt.show()


###Data cleaning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
class data_preparation:

    def cleanse_data(self, loan_data):
        loan_data = loan_data.drop('Loan_ID', axis = 1)

        self.object_cols = list(loan_data.select_dtypes(include='object'))
        self.numeric_cols =list(loan_data.select_dtypes(include=['int','float']))
        # Fill missing values with mean or mode
        for i in loan_data.columns:
            if i in self.object_cols:
                mode_value = loan_data[i].mode()
                if mode_value[0] != None:
                    loan_data[i] = loan_data[i].fillna(mode_value[0])

            elif i in self.numeric_cols:
                mean_value = loan_data[i].mean()
                if mean_value != None:
                    loan_data[i] = loan_data[i].fillna(mean_value)

        # Calculate the Total Income column
        loan_data['Total_Income'] = loan_data['ApplicantIncome'] + loan_data['CoapplicantIncome']
        self.numeric_cols.append('Total_Income')
        # Integer encoding for object columns
        for i in self.object_cols:
            var_name = "Encoded_"+str(i)

            loan_data[var_name]=pd.factorize(loan_data[i])[0]

            # Drop initial column as it is now encoded
            loan_data= loan_data.drop(i,axis = 1)

        self.cleaned_data = loan_data

    def split_data(self,seed=1,fraction=0.25):

        self.cleaned_data_train,self.cleaned_data_test=train_test_split(self.cleaned_data, test_size=0.2,random_state=seed,stratify=self.cleaned_data['Encoded_Loan_Status'])

        print('Data Shape: ',self.cleaned_data.shape)
        print('Data Train Shape: ',self.cleaned_data_train.shape)
        print('Data Test Shape: ',self.cleaned_data_test.shape)

    def fe_data(self,seed=1,scaling=True):

        self.data_train = self.cleaned_data_train.copy()
        self.data_test = self.cleaned_data_test.copy()

        # Exclude 'Credit_History' from transformation
        self.numeric_cols.remove('Credit_History')
        self.object_cols.remove('Loan_Status')

        # Log transform for numeric column
        for i in self.numeric_cols:
            var_name = "Log_"+str(i)

            # Transform amount into log
            self.data_train[var_name]=self.data_train[i].apply(lambda x: np.log(x+1))
            self.data_test[var_name]=self.data_test[i].apply(lambda x: np.log(x+1))

            # Drop initial column as it is now transformed
            self.data_train = self.data_train.drop(i,axis = 1)
            self.data_test = self.data_test.drop(i,axis = 1)

        self.normalized_data = self.data_train.copy()

        #Scaling for numeric column
        if scaling:
            self.numeric_cols = ['Log_' +str(i) for i in self.numeric_cols]

            scaler = StandardScaler()
            self.data_train[self.numeric_cols] = scaler.fit_transform(self.data_train[self.numeric_cols])
            self.data_test[self.numeric_cols] = scaler.transform (self.data_test[self.numeric_cols])

        # Make train and test dataset
        self.X_train = self.data_train.drop('Encoded_Loan_Status',axis=1)
        self.y_train = self.data_train[['Encoded_Loan_Status']]

        self.X_test = self.data_test.drop('Encoded_Loan_Status',axis=1)
        self.y_test = self.data_test['Encoded_Loan_Status']

        print(f'Data Train Label Proportion -  {self.y_train.value_counts().values}   |   Data Test Label Proportion -  {self.y_test.value_counts().values}')


# perform fill missing value, integer encoding, train-test split, log transformation, and standard scaler to the original dataset
data_prep = data_preparation() 
data_prep.cleanse_data(loan_data)
data_prep.split_data()
data_prep.fe_data(scaling=True)

# Function to detect and remove outliers using the IQR method
def remove_outliers_iqr(df, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range

    # Determine the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to exclude outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered


# List of numerical columns to check for outliers
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Iterate over each numerical column to remove outliers
for column in numerical_columns:
    print(f"Removing outliers from {column}...")
    loan_data = remove_outliers_iqr(loan_data, column)

# Check the shape of the dataset after removing outliers
print("Shape of the dataset after outlier removal:", loan_data.shape)
def plot_boxplot(column):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=loan_data[column])
    plt.title(f'Box Plot for {column}')
    plt.xlabel(column)
    plt.show()

# Plot boxplots for each numerical column
for column in numerical_columns:
    plot_boxplot(column)
def plot_distribution(column, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=loan_data, x=column, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Plot distribution for each numerical column
plot_distribution("ApplicantIncome", "Applicant Income Distribution")
plot_distribution("CoapplicantIncome", "Coapplicant Income Distribution")
plot_distribution("LoanAmount", "Loan Amount Distribution")

#SMOTE
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from collections import Counter

##Over Sampling

#SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from collections import Counter

def smote_oversampling(X,y,print_result=True):
    df_train = pd.concat([X,y],axis=1)

    smt = SMOTE(random_state = 1,sampling_strategy=0.9)
    X_train_sam, y_train_sam = smt.fit_resample(X, y)

    if print_result:
        non_default_original = df_train['Encoded_Loan_Status'].value_counts().values[0]
        default_original = df_train['Encoded_Loan_Status'].value_counts().values[1]

        non_default_sampling = y_train_sam.value_counts().values[0]
        default_sampling = y_train_sam.value_counts().values[1]

        print('Original dataset shape:', '0: ', non_default_original,'1: ', default_original)
        print('Sampling dataset shape:', '0: ', non_default_sampling,'1: ', default_sampling)
        print(f'Minority data generate: {round(100*(default_sampling-default_original)/default_original,2)}%')

    return X_train_sam, y_train_sam

X_train = data_prep.X_train
y_train = data_prep.y_train
X_train_sam, y_train_sam = smote_oversampling(X_train,y_train,print_result=True)


##Model TRainings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix


def model(models, X_train, X_test, y_train, y_test):
    models.fit(X_train, y_train)

    y_pred = models.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f'Accuracy \t - {accuracy}')
    print(f'Precision \t - 0: {round(precision[0], 2)}, 1: {round(precision[1], 2)}')
    print(f'Recall \t\t - 0: {round(recall[0], 2)}, 1: {round(recall[1], 2)}')
    print(f'F1-Score \t - 0: {round(f1[0], 2)}, 1: {round(f1[1], 2)}')
    print(f'Confusion Matrix: \n {confusion_mat}')

    return models


log_reg_model = LogisticRegression(random_state=0)
log_reg_model_eval = model(log_reg_model,data_prep.X_train,data_prep.X_test,data_prep.y_train.iloc[:,0],data_prep.y_test)

random_forest_model = RandomForestClassifier(random_state=0)
random_forest_model_eval = model(random_forest_model,data_prep.X_train,data_prep.X_test,data_prep.y_train.iloc[:,0],data_prep.y_test)


