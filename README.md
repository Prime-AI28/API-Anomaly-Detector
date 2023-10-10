# API Anomaly Detection Project

## Overview

This GitHub repository contains the source code and resources for an API anomaly detection project. The project focuses on analyzing API call data using various Python data science and machine learning algorithms, including Isolation Forest, One-Class SVM, and Random Forest. Additionally, a Streamlit dashboard has been developed to visualize and interact with the results, specifically showing data from the previous day (D-1).

Project Structure
The repository is organized as follows:

`data/`: This directory contains the dataset(s) used for anomaly detection. Please ensure you have the necessary dataset(s) before running the code.

`notebooks/`: Jupyter notebooks with code for data preprocessing, feature engineering, model training, and evaluation.

`src/`: Python scripts for modularized code used in the project.

`streamlit/`: Contains the Streamlit dashboard code and related assets for visualization and interaction.

`requirements.txt`: A list of Python dependencies required to run the project. You can use this file to set up a virtual environment.

Installation
1. Clone this repository to your local machine:
    ```git clone https://github.com/your-username/API-Anomaly-Detection.git```

2. Create a virtual environment (recommended):
    ```python -m venv venv```

3. Activate the virtual environment:

    On Windows:
        ```venv\Scripts\activate```
   
    On macOS and Linux:
        ```source venv/bin/activate```

5. Install the required Python packages:
    ```pip install -r requirements.txt```

## Usage

### Data Preparation

1. Place your API call data in the `data/` directory.

2. Use the Jupyter notebooks in the `notebooks/` directory for data preprocessing and feature engineering. Follow the instructions in the notebooks to prepare your data for modeling.

### Model Training
1. After data preparation, use the Jupyter notebooks to train the anomaly detection models (Isolation Forest, One-Class SVM, Random Forest). Evaluate the models' performance and choose the best one for your use case.

### Streamlit Dashboard
1. Navigate to the `streamlit/ directory`.

2. Run the Streamlit dashboard using the following command:
    ```streamlit run app.py```
3. Access the dashboard in your web browser (by default, it runs on `http://localhost:8501`). Explore the various graphs and visualizations, focusing on D-1 data.

## Contributing
If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix:
    ```git checkout -b feature/new-feature```

3. Make your changes and commit them:
    ```git commit -m "Add new feature"```

4. Push your changes to your fork:

    ```git push origin feature/new-feature```

5. Open a pull request to the main repository, describing your changes and their purpose.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to the open-source community and the authors of the libraries and tools used in this project.

Feel free to reach out if you have any questions or suggestions!
