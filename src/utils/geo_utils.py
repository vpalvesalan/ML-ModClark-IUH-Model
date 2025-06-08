import re

def extract_coordinates_from_point_string(string):
    """
    Extracts longitude and latitude from a string formatted as 'POINT (longitude latitude)'.
    
    Parameters:
    string (str): The input string containing the coordinates.
    
    Returns:
    tuple: A tuple containing the longitude and latitude as floats.
    """
    # Regular expression to match the POINT format
    pattern = r'POINT\s*\((.*?)\s*\)'
    
    # Match the pattern against the string
    match = re.match(pattern, string)
    
    if match:
        # Get the content inside parentheses
        coords = match.group(1).split()
        longitude = float(coords[0])
        latitude = float(coords[1])
        return longitude, latitude
    else:
        raise ValueError("No match found in the input string.")