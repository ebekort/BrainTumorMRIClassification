# Project Tests

This directory contains unit tests for the Brain Tumor MRI Classification project.

## Running Tests

The tests are written using Python's `unittest` framework. To run all tests, navigate to the root directory of the project and run the following command:

```bash
python -m unittest discover tests
```

Or, to run tests in a specific file:

```bash
python -m unittest tests.utils.test_dataset # Example for dataset tests
python -m unittest tests.utils.test_loader # Example for loader tests
python -m unittest tests.models.test_main_model # Example for main_model tests
```

## Test Structure

*   `tests/utils/test_dataset.py`: Contains unit tests for the `MRI_Dataset` class found in `project_name.utils.dataset`. These tests utilize mocking for file system operations (`os.listdir`, `os.path.join`) and image loading (`PIL.Image.open`) due to the challenges in creating reliable dummy image files in all execution environments.
*   `tests/utils/test_loader.py`: Contains unit tests for the `get_dataloaders` function in `project_name.utils.loader`. These tests mock the `MRI_Dataset` class itself to ensure `get_dataloaders` correctly handles data splitting, batching, and transform assignments.
*   `tests/models/test_main_model.py`: Contains unit tests for the `Model` class (ResNet-based model) in `project_name.models.main_model`. These tests verify model initialization, forward pass output shapes, and structural integrity (e.g., replacement of ResNet's fully connected layer).
*   `tests/test_main.py`: Currently a placeholder, can be expanded for integration tests or tests of `main.py` script functionality.

## Notes

*   Some tests rely on mocking due to limitations in creating/accessing a file-based dummy dataset in the execution environment. These tests verify the logic of data handling and processing components without requiring actual image files.
*   Ensure all dependencies from the project's `Pipfile` are installed in your environment before running tests.
