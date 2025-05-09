from time import time
import json
from functools import wraps

def measure_execution_time(func):
    """
    Decorator to measure and print the time taken to execute a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time()  # Record end time
        execution_time = end_time - start_time
        if execution_time < 60:
            print(f"\nFunction '{func.__name__}' executed in {execution_time:.4f} seconds.")
        else:
            minutes, seconds = divmod(execution_time, 60)
            print(f"\nFunction '{func.__name__}' executed in {int(minutes)} minutes and {seconds:.4f} seconds.\n")
    
        return result

    return wrapper


def process_notebook(input_path):
    """
    Load a Jupyter Notebook, clear outputs from all code cells, and save it with a modified extension.
    
    Args:
    -----
        input_path (str): Path to the input .ipynb file.
    """
    # Load the notebook
    with open(input_path, 'r', encoding='utf-8') as file:
        notebook_data = json.load(file)
    
    # Clear outputs from all code cells
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    
    # Save the updated notebook with a modified extension
    output_path = input_path.replace('.ipynb', '_cleaned.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(notebook_data, file, indent=4)
    
    print(f"Notebook cleaned and saved to: {output_path}")

