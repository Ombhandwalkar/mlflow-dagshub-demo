import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 


mlflow.set_tracking_uri('http://127.0.0.1:5000')

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
max_depth=10


mlflow.set_experiment('iris-dt')

with mlflow.start_run(experiment_id='198628815721778494',run_name='laksh'):
    dt=DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred=dt.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric('Accuracy',accuracy)
    mlflow.log_param('Max_Depth',max_depth)

    # Create Confusion matrix plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(16,12))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as an Artifact
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt,"decision tree")
    print('Accuracy',accuracy)