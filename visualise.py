import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
def app(df):
	st.header('View Data')
	with st.expander('View Dataset'):
		st.table(df)
	st.subheader('Column Description')
	if st.checkbox('Show summary'):
		st.table(df.describe())
	b1, b2, b3 = st.columns(3)
	with b1:
		if st.checkbox('Show all column name'):
			st.table(df.columns)
	with b2:
		if st.checkbox('Show all column data'):
			col2 = st.selectbox('Select column', (df.columns))
			st.write("Data Type:\t"+str(df[col2].dtype))
			st.write("Value counts:\n")
			st.table(df[col2].value_counts())
	with b3:
		if st.checkbox('Show data summary'):
			col = st.selectbox('Select columns', (df.columns))
			st.table(df[col])
	st.header('\n\nVisualize Data')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.subheader('Scatter Plot')
	ftr = st.multiselect('Select the inputs', df.columns.drop('Outcome'))
	for i in ftr:
		st.subheader(f'Scatter plot between {ftr} and Outcome')
		plt.figure(figsize=(10, 5))
		sns.scatterplot(x=i, y='Outcome', data = df, edgecolor='orange')
		st.pyplot()
		st.subheader('Visualization Selector')
		ch = st.multiselect('Select the chart', ('Histogram', 'Boxplot', 'Correlation Heatmap'))
		if 'Histogram' in ch:
			st.subheader('Histogram')
			plt.figure(figsize=(10, 5))
			plt.hist(df[i], bins='sturges', edgecolor='#f80')
			st.pyplot()
		if 'Boxplot' in ch:
			st.subheader('Boxplot')
			plt.figure(figsize=(10, 5))
			sns.boxplot(df[i], color='#f80')
			st.pyplot()
		if 'Correlation Heatmap' in ch:
			st.subheader('Correlation Heatmap')
			plt.figure(figsize=(10, 5))
			sns.heatmap(df.corr(), annot=True)
			st.pyplot()
	import warnings
	from sklearn.model_selection import train_test_split
	from sklearn.tree import DecisionTreeClassifier  
	from sklearn.model_selection import GridSearchCV  
	from sklearn import tree
	from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
	import graphviz
	from sklearn.tree import export_graphviz
	from io import StringIO
	from IPython.display import Image  


	# Define a function 'app()' which accepts 'census_df' as an input.
	def app(diabetes_df):
	    warnings.filterwarnings('ignore')
	    st.set_option('deprecation.showPyplotGlobalUse', False)
	    st.title("Visualise the Diabetes Prediction Web app ")

	    if st.checkbox("Show the correlation heatmap"):
	        st.subheader("Correlation Heatmap")
	        plt.figure(figsize = (10, 6))
	        ax = sns.heatmap(diabetes_df.iloc[:, 1:].corr(), annot = True) # Creating an object of seaborn axis and storing it in 'ax' variable
	        bottom, top = ax.get_ylim() # Getting the top and bottom margin limits.
	        ax.set_ylim(bottom + 0.5, top - 0.5) # Increasing the bottom and decreasing the top margins respectively.
	        st.pyplot()

	    st.subheader("Predictor Selection")


	    # Add a single select with label 'Select the Classifier'
	    plot_select = st.selectbox("Select the Classifier to Visualise the Diabetes Prediction:", ('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier')) 

	    if plot_select == 'Decision Tree Classifier':
	        # Split the train and test dataset. 
	        feature_columns = list(diabetes_df.columns)

	        # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
	        feature_columns.remove('Pregnancies')
	        feature_columns.remove('Skin_Thickness')
	        feature_columns.remove('Outcome')

	        X = diabetes_df[feature_columns]
	        y = diabetes_df['Outcome']
	        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

	        dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
	        dtree_clf.fit(X_train, y_train) 
	        y_train_pred = dtree_clf.predict(X_train)
	        y_test_pred = dtree_clf.predict(X_test)

	         
	        if st.checkbox("Plot confusion matrix"):
	            plt.figure(figsize = (10, 6))
	            plot_confusion_matrix(dtree_clf, X_train, y_train, values_format = 'd')
	            st.pyplot()

	        if st.checkbox("Plot Decision Tree"):   
	            # Export decision tree in dot format and store in 'dot_data' variable.
	            dot_data = tree.export_graphviz(decision_tree = dtree_clf, max_depth = 3, out_file = None, filled = True, rounded = True,
	                feature_names = feature_columns, class_names = ['0', '1'])
	            # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
	            st.graphviz_chart(dot_data)


	    if plot_select == 'GridSearchCV Best Tree Classifier':
	        # Split the train and test dataset. 
	        feature_columns = list(diabetes_df.columns)

	        # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
	        feature_columns.remove('Pregnancies')
	        feature_columns.remove('Skin_Thickness')
	        feature_columns.remove('Outcome')

	        X = diabetes_df[feature_columns]
	        y = diabetes_df['Outcome']
	        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

	        param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

	        # Create a grid
	        grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

	        # Training
	        grid_tree.fit(X_train, y_train)
	        best_tree = grid_tree.best_estimator_

	        grid_tree.fit(X_train, y_train) 
	        y_train_pred = grid_tree.predict(X_train)
	        y_test_pred = grid_tree.predict(X_test)

	         
	        if st.checkbox("Plot confusion matrix"):
	            plt.figure(figsize = (5, 3))
	            plot_confusion_matrix(grid_tree, X_train, y_train, values_format = 'd')
	            st.pyplot()

	        if st.checkbox("Plot Decision Tree"):   
	            # Create empty dot file.
	            #dot_data = StringIO()
	            # Export decision tree in dot format.
	            dot_data = tree.export_graphviz(decision_tree = best_tree, max_depth = 3, out_file = None, filled = True, rounded = True,
	                feature_names = feature_columns, class_names = ['0', '1'])
	            st.graphviz_chart(dot_data)