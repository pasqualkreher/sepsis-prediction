from pathlib import Path

def project_path():
    # Start at the current working directory
    current_path = Path.cwd()

    # Navigate upwards until finding the "mimic-iii-master" directory or stop at the filesystem root
    while current_path.parts and "mimic-iii-master" not in current_path.name:
        current_path = current_path.parent

    # If "mimic-iii-master" was not found, you may want to handle it (e.g., raise an error)
    if "mimic-iii-master" not in current_path.name:
        raise FileNotFoundError("The 'mimic-iii-master' directory was not found in any parent directories.")

    # Construct the path to the file
    project_path_file = current_path / "files" / "project_path.txt"

    # Read the path from the file
    with project_path_file.open('r') as file:
        project_path = file.read().strip()

    return project_path
