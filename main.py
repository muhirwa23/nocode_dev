# main.py

import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTabWidget, QWidget,
    QPushButton, QComboBox, QLineEdit, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import utils.data_ingestion as di
import utils.data_transformation as dt
import utils.statistical_analysis as sa
import utils.ml_models as ml
import utils.nn_utils as nn_utils
import torch

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exceptional Data Analytics Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.data = None
        self.X = None
        self.y = None
        self.model = None

        # Main tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs
        self.data_tab = QWidget()
        self.eda_tab = QWidget()
        self.analysis_tab = QWidget()
        self.ml_tab = QWidget()
        self.dashboard_tab = QWidget()

        # Adding tabs
        self.tabs.addTab(self.data_tab, "Data Management")
        self.tabs.addTab(self.eda_tab, "EDA & Visualization")
        self.tabs.addTab(self.analysis_tab, "Statistical Analysis")
        self.tabs.addTab(self.ml_tab, "Machine Learning")
        self.tabs.addTab(self.dashboard_tab, "Dashboard")

        # Setting up each tab
        self.setup_data_tab()
        self.setup_eda_tab()
        self.setup_analysis_tab()
        self.setup_ml_tab()
        self.setup_dashboard_tab()

    # ----------------- Data Management Tab -----------------
    def setup_data_tab(self):
        layout = QVBoxLayout()

        # Load Data Button
        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        # Missing Values Handling
        layout.addWidget(QLabel("Handle Missing Values:"))
        self.missing_method_dropdown = QComboBox()
        self.missing_method_dropdown.addItems(["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Custom Value", "Interpolate"])
        layout.addWidget(self.missing_method_dropdown)

        self.fill_value_input = QLineEdit()
        self.fill_value_input.setPlaceholderText("Enter custom value (if selected)")
        layout.addWidget(self.fill_value_input)

        self.apply_missing_button = QPushButton("Apply Missing Value Handling")
        self.apply_missing_button.clicked.connect(self.apply_missing_values)
        layout.addWidget(self.apply_missing_button)

        # Remove Duplicates Button
        self.remove_duplicates_button = QPushButton("Remove Duplicates")
        self.remove_duplicates_button.clicked.connect(self.remove_duplicates)
        layout.addWidget(self.remove_duplicates_button)

        # Scaling Data Button
        self.scale_button = QPushButton("Scale Data")
        self.scale_button.clicked.connect(self.scale_data)
        layout.addWidget(self.scale_button)

        # Encode Categorical Data Button
        self.encode_button = QPushButton("Encode Categorical Data")
        self.encode_button.clicked.connect(self.encode_data)
        layout.addWidget(self.encode_button)

        # Normalize Data Button
        self.normalize_button = QPushButton("Normalize Data")
        self.normalize_button.clicked.connect(self.normalize_data)
        layout.addWidget(self.normalize_button)

        # Detect and Remove Outliers Button
        self.outlier_button = QPushButton("Detect and Remove Outliers")
        self.outlier_button.clicked.connect(self.detect_remove_outliers)
        layout.addWidget(self.outlier_button)

        # Split Data Button
        self.split_button = QPushButton("Split Data into Train and Test Sets")
        self.split_button.clicked.connect(self.split_data)
        layout.addWidget(self.split_button)

        # Data Table Display
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.data_tab.setLayout(layout)

    def load_data(self):
        """Loads data from a CSV file and displays it."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.data = di.load_csv(file_path)
            if self.data is not None:
                self.display_data(self.data)
                QMessageBox.information(self, "Success", "Data loaded successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to load data.")

    def display_data(self, data):
        """Displays DataFrame in the table widget."""
        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                self.table.setItem(i, j, item)

    def apply_missing_values(self):
        """Applies selected missing value handling method."""
        if self.data is not None:
            method = self.missing_method_dropdown.currentText()
            value = self.fill_value_input.text() if method == "Fill with Custom Value" else None
            try:
                self.data = dt.handle_missing_values(self.data, method=method.lower().replace(" ", "_"), value=value)
                self.display_data(self.data)
                QMessageBox.information(self, "Success", "Missing values handled successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to handle missing values:\n{e}")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def remove_duplicates(self):
        """Removes duplicate rows from the data."""
        if self.data is not None:
            self.data = dt.remove_duplicates(self.data)
            self.display_data(self.data)
            QMessageBox.information(self, "Success", "Duplicates removed successfully!")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def scale_data(self):
        """Scales selected numerical columns."""
        if self.data is not None:
            columns, ok = QInputDialog.getText(self, "Scale Columns", "Enter columns to scale (comma-separated):")
            if ok:
                columns = [col.strip() for col in columns.split(",") if col.strip() in self.data.columns]
                if columns:
                    try:
                        self.data = dt.scale_data(self.data, columns)
                        self.display_data(self.data)
                        QMessageBox.information(self, "Success", f"Columns scaled successfully: {', '.join(columns)}")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to scale data:\n{e}")
                else:
                    QMessageBox.warning(self, "Error", "No valid columns entered.")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def encode_data(self):
        """Encodes selected categorical columns."""
        if self.data is not None:
            columns, ok = QInputDialog.getText(self, "Encode Columns", "Enter columns to encode (comma-separated):")
            if ok:
                columns = [col.strip() for col in columns.split(",") if col.strip() in self.data.columns]
                if columns:
                    try:
                        self.data = dt.encode_categorical(self.data, columns)
                        self.display_data(self.data)
                        QMessageBox.information(self, "Success", f"Columns encoded successfully: {', '.join(columns)}")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to encode data:\n{e}")
                else:
                    QMessageBox.warning(self, "Error", "No valid columns entered.")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def normalize_data(self):
        """Normalizes selected numerical columns."""
        if self.data is not None:
            columns, ok = QInputDialog.getText(self, "Normalize Columns", "Enter columns to normalize (comma-separated):")
            if ok:
                columns = [col.strip() for col in columns.split(",") if col.strip() in self.data.columns]
                if columns:
                    try:
                        self.data = dt.normalize_data(self.data, columns)
                        self.display_data(self.data)
                        QMessageBox.information(self, "Success", f"Columns normalized successfully: {', '.join(columns)}")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to normalize data:\n{e}")
                else:
                    QMessageBox.warning(self, "Error", "No valid columns entered.")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def detect_remove_outliers(self):
        """Detects and removes outliers based on Z-score threshold."""
        if self.data is not None:
            threshold, ok = QInputDialog.getDouble(self, "Outlier Threshold", "Enter Z-score threshold:", 3.0, 0.0, 10.0, 1)
            if ok:
                try:
                    self.data = dt.remove_outliers(self.data, z_threshold=threshold)
                    self.display_data(self.data)
                    QMessageBox.information(self, "Success", f"Outliers removed with Z-score threshold: {threshold}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to remove outliers:\n{e}")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    def split_data(self):
        """Splits data into training and testing sets based on the target column."""
        if self.data is not None:
            target, ok = QInputDialog.getText(self, "Target Column", "Enter target column name:")
            if ok:
                if target in self.data.columns:
                    try:
                        self.X, self.y = di.split_features_target(self.data, target)
                        QMessageBox.information(self, "Success", f"Data split successfully with target column: {target}")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to split data:\n{e}")
                else:
                    QMessageBox.warning(self, "Error", "Target column not found in data.")
        else:
            QMessageBox.warning(self, "Error", "No data loaded.")

    # ----------------- EDA & Visualization Tab -----------------
    def setup_eda_tab(self):
        layout = QVBoxLayout()

        # Chart Type Selection
        layout.addWidget(QLabel("Choose a Seaborn Chart Type:"))
        self.chart_dropdown = QComboBox()
        self.chart_dropdown.addItems([
            "Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Histogram",
            "Violin Plot", "Pair Plot", "Heatmap", "Swarm Plot", "Count Plot"
        ])
        layout.addWidget(self.chart_dropdown)

        # Create Chart Button
        self.create_chart_button = QPushButton("Create Chart")
        self.create_chart_button.clicked.connect(self.create_chart)
        layout.addWidget(self.create_chart_button)

        self.eda_tab.setLayout(layout)

    def create_chart(self):
        """Creates a Seaborn chart based on user selection."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        chart_type = self.chart_dropdown.currentText()

        # Get X-axis column
        x_col, ok_x = QInputDialog.getText(self, "X-Axis", "Enter X-axis column name:")
        if not ok_x or x_col not in self.data.columns:
            QMessageBox.warning(self, "Error", "Invalid or missing X-axis column.")
            return

        # Get Y-axis column (optional)
        y_col, ok_y = QInputDialog.getText(self, "Y-Axis", "Enter Y-axis column name (optional):")
        y_col = y_col if ok_y and y_col in self.data.columns else None

        # Get Hue column (optional)
        hue_col, ok_hue = QInputDialog.getText(self, "Hue", "Enter Hue column name (optional):")
        hue = hue_col if ok_hue and hue_col in self.data.columns else None

        plt.figure(figsize=(10, 6))
        try:
            if chart_type == "Scatter Plot":
                if y_col:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    QMessageBox.warning(self, "Error", "Y-axis column is required for Scatter Plot.")
                    return
            elif chart_type == "Line Plot":
                if y_col:
                    sns.lineplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    QMessageBox.warning(self, "Error", "Y-axis column is required for Line Plot.")
                    return
            elif chart_type == "Bar Plot":
                if y_col:
                    sns.barplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    sns.barplot(data=self.data, x=x_col, y=x_col, hue=hue)
            elif chart_type == "Box Plot":
                if y_col:
                    sns.boxplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    sns.boxplot(data=self.data, x=x_col, y=x_col, hue=hue)
            elif chart_type == "Histogram":
                sns.histplot(data=self.data, x=x_col, hue=hue, kde=True)
            elif chart_type == "Violin Plot":
                if y_col:
                    sns.violinplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    QMessageBox.warning(self, "Error", "Y-axis column is required for Violin Plot.")
                    return
            elif chart_type == "Pair Plot":
                sns.pairplot(self.data, hue=hue)
            elif chart_type == "Heatmap":
                sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
            elif chart_type == "Swarm Plot":
                if y_col:
                    sns.swarmplot(data=self.data, x=x_col, y=y_col, hue=hue)
                else:
                    QMessageBox.warning(self, "Error", "Y-axis column is required for Swarm Plot.")
                    return
            elif chart_type == "Count Plot":
                sns.countplot(data=self.data, x=x_col, hue=hue)
            plt.title(f"{chart_type} of {x_col}" + (f" and {y_col}" if y_col else ""))
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create chart:\n{e}")

    # ----------------- Statistical Analysis Tab -----------------
    def setup_analysis_tab(self):
        layout = QVBoxLayout()

        # Statistical Test Selection
        layout.addWidget(QLabel("Choose a Statistical Test:"))
        self.stat_test_dropdown = QComboBox()
        self.stat_test_dropdown.addItems([
            "T-Test", "Chi-Square Test", "ANOVA", "Linear Regression", 
            "Logistic Regression"
        ])
        layout.addWidget(self.stat_test_dropdown)

        # Run Test Button
        self.run_test_button = QPushButton("Run Test")
        self.run_test_button.clicked.connect(self.run_stat_test)
        layout.addWidget(self.run_test_button)

        # Display Results
        self.result_label = QLabel("Results will appear here.")
        layout.addWidget(self.result_label)

        self.analysis_tab.setLayout(layout)

    def run_stat_test(self):
        """Runs the selected statistical test."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        test = self.stat_test_dropdown.currentText()

        try:
            if test == "T-Test":
                col1, ok1 = QInputDialog.getText(self, "T-Test", "Enter first column name:")
                col2, ok2 = QInputDialog.getText(self, "T-Test", "Enter second column name:")
                if ok1 and ok2:
                    result = sa.t_test(self.data, col1, col2)
                    self.result_label.setText(f"T-Test Result:\nStatistic: {result.statistic}\nP-Value: {result.pvalue}")
            elif test == "Chi-Square Test":
                col1, ok1 = QInputDialog.getText(self, "Chi-Square Test", "Enter first categorical column name:")
                col2, ok2 = QInputDialog.getText(self, "Chi-Square Test", "Enter second categorical column name:")
                if ok1 and ok2:
                    result = sa.chi_square_test(self.data, col1, col2)
                    self.result_label.setText(f"Chi-Square Test Result:\nChi2: {result[0]}\nP-Value: {result[1]}\nDoF: {result[2]}\nExpected Frequencies:\n{result[3]}")
            elif test == "ANOVA":
                dependent_var, ok_dep = QInputDialog.getText(self, "ANOVA", "Enter dependent variable column name:")
                independent_var, ok_ind = QInputDialog.getText(self, "ANOVA", "Enter independent variable column name:")
                if ok_dep and ok_ind:
                    result = sa.anova_test(self.data, dependent_var, independent_var)
                    self.result_label.setText(f"ANOVA Test Result:\n{result}")
            elif test == "Linear Regression":
                target, ok_target = QInputDialog.getText(self, "Linear Regression", "Enter target column name:")
                if ok_target and target in self.data.columns:
                    X, y = di.split_features_target(self.data, target)
                    summary = sa.linear_regression(X, y)
                    self.display_text(summary.as_text())
                else:
                    QMessageBox.warning(self, "Error", "Invalid target column.")
            elif test == "Logistic Regression":
                target, ok_target = QInputDialog.getText(self, "Logistic Regression", "Enter target column name:")
                if ok_target and target in self.data.columns:
                    X, y = di.split_features_target(self.data, target)
                    summary = sa.logistic_regression(X, y)
                    self.display_text(summary.as_text())
                else:
                    QMessageBox.warning(self, "Error", "Invalid target column.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to run test:\n{e}")

    def display_text(self, text):
        """Displays text in a message box."""
        msg = QMessageBox()
        msg.setWindowTitle("Test Results")
        msg.setText(text)
        msg.exec_()

    # ----------------- Machine Learning Tab -----------------
    def setup_ml_tab(self):
        layout = QVBoxLayout()

        # Model Selection
        layout.addWidget(QLabel("Choose a Machine Learning Model:"))
        self.ml_model_dropdown = QComboBox()
        self.ml_model_dropdown.addItems([
            "Linear Regression", "Logistic Regression", "K-Means Clustering", 
            "Neural Network (PyTorch)"
        ])
        layout.addWidget(self.ml_model_dropdown)

        # Train Model Button
        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_model_button)

        # Display Model Results
        self.ml_result_label = QLabel("Model results will appear here.")
        layout.addWidget(self.ml_result_label)

        self.ml_tab.setLayout(layout)

    def train_model(self):
        """Trains the selected machine learning model."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load and prepare the data first.")
            return

        model_type = self.ml_model_dropdown.currentText()

        try:
            if model_type == "Linear Regression":
                if self.X is not None and self.y is not None:
                    model = ml.train_linear_regression(self.X, self.y)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    mse = ml.evaluate_regression(model, X_test, y_test)
                    self.ml_result_label.setText(f"Linear Regression MSE: {mse:.4f}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
            elif model_type == "Logistic Regression":
                if self.X is not None and self.y is not None:
                    model = ml.train_logistic_regression(self.X, self.y)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    accuracy, cm = ml.evaluate_classification(model, X_test, y_test)
                    self.ml_result_label.setText(f"Logistic Regression Accuracy: {accuracy:.4f}\nConfusion Matrix:\n{cm}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
            elif model_type == "K-Means Clustering":
                n_clusters, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter number of clusters:", min=1, value=3)
                if ok:
                    model = ml.train_kmeans(self.X, n_clusters)
                    self.data['Cluster'] = model.labels_
                    self.display_data(self.data)
                    self.ml_result_label.setText(f"K-Means Clustering trained with {n_clusters} clusters.")
            elif model_type == "Neural Network (PyTorch)":
                if self.X is not None and self.y is not None:
                    # Prepare data
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    model = nn_utils.train_neural_network(X_train, y_train, epochs=100, learning_rate=0.001)
                    # Simple evaluation
                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
                        outputs = model(X_test_tensor).squeeze()
                        if len(outputs.shape) > 1:
                            outputs = torch.argmax(outputs, dim=1)
                            y_pred = torch.argmax(outputs, dim=1)
                        else:
                            y_pred = (outputs > 0.5).float()
                        if self.y.nunique() == 2:
                            accuracy = (y_pred == y_test_tensor).sum().item() / len(y_test_tensor)
                            self.ml_result_label.setText(f"Neural Network Accuracy: {accuracy:.4f}")
                        else:
                            mse = torch.mean((outputs - y_test_tensor) ** 2).item()
                            self.ml_result_label.setText(f"Neural Network MSE: {mse:.4f}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to train model:\n{e}")

    # ----------------- Dashboard Tab -----------------
    def setup_dashboard_tab(self):
        layout = QVBoxLayout()

        # Component Palette
        self.component_palette = QListWidget()
        self.component_palette.setFixedWidth(200)
        self.component_palette.setDragEnabled(True)
        self.component_palette.addItems([
            "Histogram", "Boxplot", "Scatter Plot", "Heatmap",
            "Pair Plot", "Violin Plot", "Swarm Plot", "Count Plot",
            "Survival Analysis"
        ])
        layout.addWidget(QLabel("Drag Components to Dashboard:"))
        layout.addWidget(self.component_palette)

        # Dashboard Workspace
        self.dashboard_scene = QGraphicsScene()
        self.dashboard_view = QGraphicsView(self.dashboard_scene)
        self.dashboard_view.setAcceptDrops(True)
        layout.addWidget(self.dashboard_view)

        self.dashboard_tab.setLayout(layout)

    def dragEnterEvent(self, event):
        """Accept drag event."""
        event.accept()

    def dropEvent(self, event):
        """Handle drop event to add components to the dashboard."""
        component = event.mimeData().text()
        if component == "Histogram":
            self.create_dashboard_chart("Histogram")
        elif component == "Boxplot":
            self.create_dashboard_chart("Boxplot")
        elif component == "Scatter Plot":
            self.create_dashboard_chart("Scatter Plot")
        elif component == "Heatmap":
            self.create_dashboard_chart("Heatmap")
        elif component == "Pair Plot":
            self.create_dashboard_chart("Pair Plot")
        elif component == "Violin Plot":
            self.create_dashboard_chart("Violin Plot")
        elif component == "Swarm Plot":
            self.create_dashboard_chart("Swarm Plot")
        elif component == "Count Plot":
            self.create_dashboard_chart("Count Plot")
        elif component == "Survival Analysis":
            self.create_dashboard_chart("Survival Analysis")
        event.accept()

    def create_dashboard_chart(self, chart_type):
        """Creates and adds a chart to the dashboard."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        try:
            plt.figure(figsize=(6, 4))
            if chart_type == "Histogram":
                x_col, ok = QInputDialog.getText(self, "Histogram", "Enter column name:")
                if ok and x_col in self.data.columns:
                    sns.histplot(self.data[x_col], kde=True)
            elif chart_type == "Boxplot":
                x_col, ok_x = QInputDialog.getText(self, "Boxplot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Boxplot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.boxplot(data=self.data, x=x_col, y=y_col)
            elif chart_type == "Scatter Plot":
                x_col, ok_x = QInputDialog.getText(self, "Scatter Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Scatter Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col)
            elif chart_type == "Heatmap":
                sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
            elif chart_type == "Pair Plot":
                sns.pairplot(self.data)
            elif chart_type == "Violin Plot":
                x_col, ok_x = QInputDialog.getText(self, "Violin Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Violin Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.violinplot(data=self.data, x=x_col, y=y_col)
            elif chart_type == "Swarm Plot":
                x_col, ok_x = QInputDialog.getText(self, "Swarm Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Swarm Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.swarmplot(data=self.data, x=x_col, y=y_col)
            elif chart_type == "Count Plot":
                x_col, ok = QInputDialog.getText(self, "Count Plot", "Enter X-axis column name:")
                if ok and x_col in self.data.columns:
                    sns.countplot(data=self.data, x=x_col)
            elif chart_type == "Survival Analysis":
                duration_col, ok_dur = QInputDialog.getText(self, "Survival Analysis", "Enter duration column name:")
                event_col, ok_event = QInputDialog.getText(self, "Survival Analysis", "Enter event column name:")
                if ok_dur and ok_event and duration_col in self.data.columns and event_col in self.data.columns:
                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    kmf.fit(durations=self.data[duration_col], event_observed=self.data[event_col])
                    kmf.plot_survival_function()
            plt.title(chart_type)
            plt.tight_layout()
            plt.savefig("temp_chart.png")
            plt.close()

            # Add the image to the dashboard
            pixmap = QPixmap("temp_chart.png")
            self.dashboard_scene.addPixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create chart:\n{e}")

    # ----------------- Statistical Analysis Tab -----------------
    def setup_analysis_tab(self):
        layout = QVBoxLayout()

        # Statistical Test Selection
        layout.addWidget(QLabel("Choose a Statistical Test:"))
        self.stat_test_dropdown = QComboBox()
        self.stat_test_dropdown.addItems([
            "T-Test", "Chi-Square Test", "ANOVA", "Linear Regression", 
            "Logistic Regression"
        ])
        layout.addWidget(self.stat_test_dropdown)

        # Run Test Button
        self.run_test_button = QPushButton("Run Test")
        self.run_test_button.clicked.connect(self.run_stat_test)
        layout.addWidget(self.run_test_button)

        # Display Results
        self.result_label = QLabel("Results will appear here.")
        layout.addWidget(self.result_label)

        self.analysis_tab.setLayout(layout)

    def run_stat_test(self):
        """Runs the selected statistical test."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        test = self.stat_test_dropdown.currentText()

        try:
            if test == "T-Test":
                col1, ok1 = QInputDialog.getText(self, "T-Test", "Enter first column name:")
                col2, ok2 = QInputDialog.getText(self, "T-Test", "Enter second column name:")
                if ok1 and ok2:
                    result = sa.t_test(self.data, col1, col2)
                    self.result_label.setText(f"T-Test Result:\nStatistic: {result.statistic}\nP-Value: {result.pvalue}")
            elif test == "Chi-Square Test":
                col1, ok1 = QInputDialog.getText(self, "Chi-Square Test", "Enter first categorical column name:")
                col2, ok2 = QInputDialog.getText(self, "Chi-Square Test", "Enter second categorical column name:")
                if ok1 and ok2:
                    result = sa.chi_square_test(self.data, col1, col2)
                    self.result_label.setText(f"Chi-Square Test Result:\nChi2: {result[0]}\nP-Value: {result[1]}\nDoF: {result[2]}\nExpected Frequencies:\n{result[3]}")
            elif test == "ANOVA":
                dependent_var, ok_dep = QInputDialog.getText(self, "ANOVA", "Enter dependent variable column name:")
                independent_var, ok_ind = QInputDialog.getText(self, "ANOVA", "Enter independent variable column name:")
                if ok_dep and ok_ind:
                    result = sa.anova_test(self.data, dependent_var, independent_var)
                    self.result_label.setText(f"ANOVA Test Result:\n{result}")
            elif test == "Linear Regression":
                target, ok_target = QInputDialog.getText(self, "Linear Regression", "Enter target column name:")
                if ok_target and target in self.data.columns:
                    X, y = di.split_features_target(self.data, target)
                    summary = sa.linear_regression(X, y)
                    self.display_text(summary.as_text())
                else:
                    QMessageBox.warning(self, "Error", "Invalid target column.")
            elif test == "Logistic Regression":
                target, ok_target = QInputDialog.getText(self, "Logistic Regression", "Enter target column name:")
                if ok_target and target in self.data.columns:
                    X, y = di.split_features_target(self.data, target)
                    summary = sa.logistic_regression(X, y)
                    self.display_text(summary.as_text())
                else:
                    QMessageBox.warning(self, "Error", "Invalid target column.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to run test:\n{e}")

    def display_text(self, text):
        """Displays text in a message box."""
        msg = QMessageBox()
        msg.setWindowTitle("Test Results")
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    # ----------------- Machine Learning Tab -----------------
    def setup_ml_tab(self):
        layout = QVBoxLayout()

        # Model Selection
        layout.addWidget(QLabel("Choose a Machine Learning Model:"))
        self.ml_model_dropdown = QComboBox()
        self.ml_model_dropdown.addItems([
            "Linear Regression", "Logistic Regression", "K-Means Clustering", 
            "Neural Network (PyTorch)"
        ])
        layout.addWidget(self.ml_model_dropdown)

        # Train Model Button
        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_model_button)

        # Display Model Results
        self.ml_result_label = QLabel("Model results will appear here.")
        layout.addWidget(self.ml_result_label)

        self.ml_tab.setLayout(layout)

    def train_model(self):
        """Trains the selected machine learning model."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load and prepare the data first.")
            return

        model_type = self.ml_model_dropdown.currentText()

        try:
            if model_type == "Linear Regression":
                if self.X is not None and self.y is not None:
                    model = ml.train_linear_regression(self.X, self.y)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    mse = ml.evaluate_regression(model, X_test, y_test)
                    self.ml_result_label.setText(f"Linear Regression MSE: {mse:.4f}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
            elif model_type == "Logistic Regression":
                if self.X is not None and self.y is not None:
                    model = ml.train_logistic_regression(self.X, self.y)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    accuracy, cm = ml.evaluate_classification(model, X_test, y_test)
                    self.ml_result_label.setText(f"Logistic Regression Accuracy: {accuracy:.4f}\nConfusion Matrix:\n{cm}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
            elif model_type == "K-Means Clustering":
                n_clusters, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter number of clusters:", min=1, value=3)
                if ok:
                    model = ml.train_kmeans(self.X, n_clusters)
                    self.data['Cluster'] = model.labels_
                    self.display_data(self.data)
                    self.ml_result_label.setText(f"K-Means Clustering trained with {n_clusters} clusters.")
            elif model_type == "Neural Network (PyTorch)":
                if self.X is not None and self.y is not None:
                    # Prepare data
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                    model = nn_utils.train_neural_network(X_train, y_train, epochs=100, learning_rate=0.001)
                    # Simple evaluation
                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
                        outputs = model(X_test_tensor).squeeze()
                        if self.y.nunique() == 2:
                            y_pred = (outputs > 0.5).float()
                            accuracy = (y_pred == y_test_tensor).sum().item() / len(y_test_tensor)
                            self.ml_result_label.setText(f"Neural Network Accuracy: {accuracy:.4f}")
                        else:
                            mse = torch.mean((outputs - y_test_tensor) ** 2).item()
                            self.ml_result_label.setText(f"Neural Network MSE: {mse:.4f}")
                else:
                    QMessageBox.warning(self, "Error", "Please split the data first.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to train model:\n{e}")

    # ----------------- Dashboard Tab -----------------
    def setup_dashboard_tab(self):
        layout = QVBoxLayout()

        # Component Palette
        self.component_palette = QListWidget()
        self.component_palette.setFixedWidth(200)
        self.component_palette.setDragEnabled(True)
        self.component_palette.addItems([
            "Histogram", "Boxplot", "Scatter Plot", "Heatmap",
            "Pair Plot", "Violin Plot", "Swarm Plot", "Count Plot",
            "Survival Analysis"
        ])
        layout.addWidget(QLabel("Drag Components to Dashboard:"))
        layout.addWidget(self.component_palette)

        # Dashboard Workspace
        self.dashboard_scene = QGraphicsScene()
        self.dashboard_view = QGraphicsView(self.dashboard_scene)
        self.dashboard_view.setAcceptDrops(True)
        layout.addWidget(self.dashboard_view)

        self.dashboard_tab.setLayout(layout)

    def dragEnterEvent(self, event):
        """Accept drag event."""
        event.accept()

    def dropEvent(self, event):
        """Handle drop event to add components to the dashboard."""
        component = event.mimeData().text()
        if component == "Histogram":
            self.create_dashboard_chart("Histogram")
        elif component == "Boxplot":
            self.create_dashboard_chart("Boxplot")
        elif component == "Scatter Plot":
            self.create_dashboard_chart("Scatter Plot")
        elif component == "Heatmap":
            self.create_dashboard_chart("Heatmap")
        elif component == "Pair Plot":
            self.create_dashboard_chart("Pair Plot")
        elif component == "Violin Plot":
            self.create_dashboard_chart("Violin Plot")
        elif component == "Swarm Plot":
            self.create_dashboard_chart("Swarm Plot")
        elif component == "Count Plot":
            self.create_dashboard_chart("Count Plot")
        elif component == "Survival Analysis":
            self.create_dashboard_chart("Survival Analysis")
        event.accept()

    def create_dashboard_chart(self, chart_type):
        """Creates and adds a chart to the dashboard."""
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        try:
            plt.figure(figsize=(6, 4))
            if chart_type == "Histogram":
                x_col, ok = QInputDialog.getText(self, "Histogram", "Enter column name:")
                if ok and x_col in self.data.columns:
                    sns.histplot(self.data[x_col], kde=True)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column name.")
                    return
            elif chart_type == "Boxplot":
                x_col, ok_x = QInputDialog.getText(self, "Boxplot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Boxplot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.boxplot(data=self.data, x=x_col, y=y_col)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column names.")
                    return
            elif chart_type == "Scatter Plot":
                x_col, ok_x = QInputDialog.getText(self, "Scatter Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Scatter Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.scatterplot(data=self.data, x=x_col, y=y_col)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column names.")
                    return
            elif chart_type == "Heatmap":
                sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
            elif chart_type == "Pair Plot":
                sns.pairplot(self.data)
            elif chart_type == "Violin Plot":
                x_col, ok_x = QInputDialog.getText(self, "Violin Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Violin Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.violinplot(data=self.data, x=x_col, y=y_col)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column names.")
                    return
            elif chart_type == "Swarm Plot":
                x_col, ok_x = QInputDialog.getText(self, "Swarm Plot", "Enter X-axis column name:")
                y_col, ok_y = QInputDialog.getText(self, "Swarm Plot", "Enter Y-axis column name:")
                if ok_x and ok_y and x_col in self.data.columns and y_col in self.data.columns:
                    sns.swarmplot(data=self.data, x=x_col, y=y_col)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column names.")
                    return
            elif chart_type == "Count Plot":
                x_col, ok = QInputDialog.getText(self, "Count Plot", "Enter X-axis column name:")
                if ok and x_col in self.data.columns:
                    sns.countplot(data=self.data, x=x_col)
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column name.")
                    return
            elif chart_type == "Survival Analysis":
                duration_col, ok_dur = QInputDialog.getText(self, "Survival Analysis", "Enter duration column name:")
                event_col, ok_event = QInputDialog.getText(self, "Survival Analysis", "Enter event column name:")
                if ok_dur and ok_event and duration_col in self.data.columns and event_col in self.data.columns:
                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    kmf.fit(durations=self.data[duration_col], event_observed=self.data[event_col])
                    kmf.plot_survival_function()
                else:
                    QMessageBox.warning(self, "Error", "Invalid or missing column names.")
                    return
            else:
                QMessageBox.warning(self, "Error", "Unsupported chart type.")
                return

            plt.title(chart_type)
            plt.tight_layout()
            plt.savefig("temp_chart.png")
            plt.close()

            # Add the image to the dashboard
            pixmap = QPixmap("temp_chart.png")
            self.dashboard_scene.addPixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create chart:\n{e}")

    # ----------------- Application Execution -----------------
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?', 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
