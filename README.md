# Classification-model-pipeline

- 1. Imports
Various libraries are imported like yaml, importlib, numpy, and sklearn.metrics.
importlib allows for dynamic importing of modules based on strings (e.g., class or module names).
yaml is used to read the YAML configuration file that contains model and parameter details.
logger is for logging operations to track progress.
config contains classes like MetricInfoArtifact, InitializeModelDetails, RandomSearchBestModel, and BestModel.
- 2. Constructor ```__init__```
This method initializes the class by reading the model configuration file (model.yaml).
It extracts details about the models and parameters for Random Search Cross-Validation from the config.
Key variables initialized:
self.random_search_cv_module: Stores the module name for RandomizedSearchCV.
self.random_search_class_name: Stores the class name for RandomizedSearchCV.
self.random_search_property_data: Stores parameter properties for Random Search.
self.models_initialization_config: Stores details about models and their initialization configurations.
- 3. read_params Method
This static method reads the YAML configuration file and returns the data in a dictionary format.
- 4. update_property_of_class Method
This method dynamically sets properties for an object using setattr. It assigns values to the properties of models such as fit_intercept.
- 5. class_for_name Method
This method dynamically imports a module and class by name. For example, it can import LogisticRegression from sklearn.linear_model using a string.
- 6. execute_random_search_operation Method
This method performs the random search operation on a model to find the best parameters.
If Random Search parameters are not provided, it directly trains the model without optimization.
If Random Search is required, it initializes the Random Search CV, applies the parameter search, and fits the model to the input features.
- 7. get_initialized_model_list Method
This method generates a list of initialized models by dynamically importing them and setting any specified parameters.
It reads model details from the configuration file, initializes models, and applies configurations such as fitting intercepts if specified.
- 8. initiate_best_parameter_search_for_initialized_model Method
This method initiates the best parameter search for a given model using Random Search.
- 9. initiate_best_parameter_search_for_initialized_models Method
This method loops through all initialized models and applies Random Search on each to find the best-performing model based on specified criteria.
- 10. get_model_detail Method
This static method retrieves details of a specific model based on its serial number from the list of initialized models.
- 11. get_best_model_from_random_searched_best_model_list Method
After all models have undergone Random Search, this method compares their scores against a baseline accuracy (default: 0.6) and returns the best-performing model.
- 12. get_best_model Method
This is the main method for executing the full model selection process. It:
Initializes models from the config file.
Performs Random Search to optimize model parameters.
Returns the best model that meets or exceeds the baseline accuracy.
Overall Workflow:
# Initialization:

The ModelFinder is initialized with the path to the configuration file (model.yaml), which contains the details for models and their parameter search space.
Model Initialization:

It dynamically imports and initializes models based on the config, setting properties like intercepts if needed.
Random Search:

It uses RandomSearchCV to search for the best hyperparameters for each model, unless specific parameters are not provided, in which case it just trains the model directly.
Model Selection:

After training, the models are compared based on their performance, and the best one (with accuracy > 0.6) is selected and returned.