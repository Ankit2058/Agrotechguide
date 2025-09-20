# Agrotechguide

Agrotechguide is a crop recommendation prototype that combines agronomic domain knowledge with machine learning to help farmers decide which crop to plant under a given set of soil and weather conditions. The repository contains the training data, experimentation notebooks, model assets, and a lightweight Flask API that exposes the recommendation engine for integration with other applications.

## Key Features
- **Curated agronomic dataset** – `agrotech.csv` provides 2,200 observations with macro-nutrient levels (N, P, K), weather readings (temperature, humidity, rainfall), soil pH, and the historically successful crop label for each scenario.
- **Machine learning workflows** – Scripts such as `atg.2.py` and `machine learning/machine_learning.py` perform label encoding, data splitting, RandomForest training, hyper-parameter search, and persistence of the fitted estimator with `joblib`/`pickle` for reuse.
- **REST integration** – The Flask app under `machine learning/machine_learning.py` loads the trained model and responds to `/api/send_data` POST requests with the recommended crop, making it easy to connect with front-end or mobile clients.
- **Supporting documentation** – The `documentation/` directory hosts reports, presentations, and imagery that describe the wider project context and deliverables for academic submission.

## Repository Structure
```
.
├── agrotech.csv                # Training dataset used by the models
├── atg.2.py                    # Stand-alone training & inference script for RandomForest
├── machine learning/
│   ├── machine_learning.py     # Flask API plus model training utilities
│   ├── data_transfer.py        # Minimal Flask prototype for data exchange
│   ├── new_rf_model.pickle     # Sample trained RandomForest model
│   ├── features_data.json      # Feature column metadata for inference
│   └── Agrotechguidedata.csv   # Additional dataset used during experimentation
├── documentation/              # Project report, presentation, and reference material
├── *.ipynb                     # Jupyter notebooks with exploratory analyses
└── README.md                   # Project overview and setup instructions (this file)
```

## Getting Started
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agrotechguide
   ```
2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies**
   The Python scripts rely on the following packages: `numpy`, `pandas`, `scikit-learn`, `flask`, `requests`, `matplotlib`, `seaborn`, and `joblib`. Install them with:
   ```bash
   pip install numpy pandas scikit-learn flask requests matplotlib seaborn joblib
   ```

## Training or Updating the Model
- The quickest way to (re-)train and test the model is to execute `atg.2.py`. Update the path in `pd.read_csv(...)` if necessary so that it points to the `agrotech.csv` file bundled with this repository. The script will automatically train a `RandomForestClassifier`, tune a subset of its hyper-parameters with `RandomizedSearchCV`, persist the best model to `agrotech_model.pkl`, and print a sample prediction.
- For more granular experimentation (visualisations, diagnostics, and saving `new_rf_model.pickle`/`features_data.json`), work with `machine learning/machine_learning.py`. This file includes helper functions for exploratory data analysis, RandomForest tuning, and dataset inspection that were used during development. Update the dataset path near the top of the file before running the training utilities.

## Serving Recommendations via the API
1. Ensure the trained model artefacts (`new_rf_model.pickle` and `features_data.json`) are present in `machine learning/`.
2. Launch the Flask service:
   ```bash
   cd "machine learning"
   python machine_learning.py
   ```
3. Issue a POST request to the `/api/send_data` endpoint with the required features. Example payload:
   ```json
   {
     "temperature": 30,
     "humidity": 60,
     "region": 31
   }
   ```
   The API will call the `last_part` helper to assemble the feature vector, perform inference with the trained RandomForest model, and respond with `{ "recommended_crop": "<crop-name>" }`.

## Data Dictionary
| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| `N`          | Nitrogen concentration in the soil (kg/ha)       |
| `P`          | Phosphorus concentration in the soil (kg/ha)     |
| `K`          | Potassium concentration in the soil (kg/ha)      |
| `temperature`| Average ambient temperature (°C)                 |
| `humidity`   | Relative humidity (%)                            |
| `ph`         | Soil pH level                                    |
| `rainfall`   | Rainfall during the growing season (mm)          |
| `label`      | Recommended crop for the given conditions        |

## Additional Resources
- See `documentation/` for full project reports and presentation material.
- Review the Jupyter notebooks in the root and `machine learning/` directories for exploratory analysis and experimentation history.

## Contributing
Contributions that improve the dataset, expand the model suite, or enhance the API are welcome. Please fork the repository, make your changes on a feature branch, and submit a pull request describing your updates. Remember to document any new functionality and include reproducible steps for reviewers.

## License
This repository currently does not declare an explicit software license. If you plan to use the code or data outside of personal experimentation, please contact the original authors or maintainers to confirm usage rights.
