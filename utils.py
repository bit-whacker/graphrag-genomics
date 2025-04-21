import streamlit as st
import os
import yaml
import shutil
import argparse
from PyPDF2 import PdfReader
import subprocess
from datetime import datetime


def init_index_proj(input_dir):
    os.system(f"bash cmd_graphrag.bat {input_dir}")
    return True

def init_index_proj2(input_dir):

    # Path to the batch file
    batch_file = "cmd_graphrag.bat"

    # Input directory to pass as an argument
    #input_dir = "C:\\path\\to\\inputDir"

    # Execute the batch script with the input directory as an argument
    try:
        result = subprocess.run(
            [batch_file, input_dir],  # Command and arguments
            shell=True,               # Required for executing batch files on Windows
            check=True,               # Raise an exception if the command fails
            stdout=subprocess.PIPE,   # Capture standard output
            stderr=subprocess.PIPE,   # Capture standard error
            text=True                 # Decode bytes to strings
        )
        
        # Print the output of the batch script
        print("Script output:")
        print(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the batch script.")
        print("Exit code:", e.returncode)
        print("Error message:", e.stderr)

    return False


def load_project_names(parent_dir = "./"):
    #st.info("loading ...")
    try:
        subdirectories = [
            name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name)) and name.startswith('proj_')
        ]
        #st.write(f"Subdirectories starting with 'case_files': {subdirectories}")
        return subdirectories
    except Exception as e:
        print(f"Error while reading existing projects: {e}")
        return []

def prefix_project(project_name):
    return f"proj_{project_name}"


def update_yaml_property(file_path, kv_dic):
    """
    Updates a specific property or sub-property in a YAML file.

    :param file_path: Path to the YAML file.
    :param property_path: Dot-separated string representing the property to update (e.g., 'key1.key2.key3').
    :param new_value: The new value to set for the specified property.
    """

    try:
        # Load the YAML file
        with open(file_path, 'r') as file:
            settings = yaml.safe_load(file)

        for d_key, d_value in kv_dic.items():
            # Navigate to the specified property
            keys = d_key.split('.')
            current = settings
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}  # Create nested dictionaries if missing
                current = current[key]

            # Update the value
            current[keys[-1]] = d_value
            st.info(f"Updated '{d_key}' to {d_value} in {file_path}")

        # Save the updated YAML file
        with open(file_path, 'w') as file:
            yaml.dump(settings, file, default_flow_style=False)

        

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        st.error(f"Error: The file '{file_path}' does not exist.")
    except yaml.YAMLError as e:
        print(f"Error processing YAML file: {e}")
        st.error(f"Error processing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        st.error(f"An unexpected error occurred: {e}")

def copy_folder_contents(source_folder, destination_folder):
    """
    Copies all contents from the source folder to the destination folder.
    Creates the destination folder if it does not exist.

    :param source_folder: Path to the source folder.
    :param destination_folder: Path to the destination folder.
    """

    st.write(f"copying {source_folder} to {destination_folder}")
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)  # Create the destination folder

        for item in os.listdir(source_folder):
            source_item = os.path.join(source_folder, item)
            destination_item = os.path.join(destination_folder, item)

            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item)
            else:
                shutil.copy2(source_item, destination_item)

        st.write(f"Contents of '{source_folder}' successfully copied to '{destination_folder}'.")

    except FileNotFoundError:
        print(f"Error: The source folder '{source_folder}' does not exist.")
        st.error(f"Error: The source folder '{source_folder}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied while accessing '{source_folder}' or '{destination_folder}'.")
        st.error(f"Error: Permission denied while accessing '{source_folder}' or '{destination_folder}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        st.error(f"An unexpected error occurred: {e}")


def extract_text_from_pdfs(pdf_path, output_dir):

    filename = pdf_path.name
    text_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

    try:
        # Read the PDF and extract text
        reader = PdfReader(pdf_path)
        extracted_text = ""

        for page in reader.pages:
            extracted_text += page.extract_text()

        # Save the extracted text to a text file
        with open(text_output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(extracted_text)

        st.write(f"Extracted text from '{filename}' and saved to '{text_output_path}'")

    except Exception as e:
        print(f"Failed to process '{filename}': {e}")
        st.error(f"Failed to process '{filename}': {e}")


def create_project_dir(proj_dir):
    if not os.path.exists(proj_dir):
            os.makedirs(proj_dir)
    else:
        st.error(f"directory - {proj_dir} already exists!")



def get_latest_folder(directory):
    st.write(directory)
    """
    Gets the most recent folder from the specified directory based on the naming convention 'YYYYMMDD-HHMMSS'.

    Parameters:
        directory (str): The path to the directory containing subdirectories.

    Returns:
        str: The name of the most recent folder, or None if no valid folders are found.
    """
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        #st.write(entries)
        # Filter for directories with the correct naming convention
        valid_folders = []
        for entry in entries:
            if os.path.isdir(os.path.join(directory, entry)):
                #st.info(entry)
                parts = entry.split('-')
                if len(parts) == 2:
                    date_part, time_part = parts
                    try:
                        #st.warning(f"{date_part} -- {time_part}")
                        # Validate the date and time format
                        datetime.strptime(date_part + time_part, '%Y%m%d%H%M%S')
                        valid_folders.append(entry)
                    except ValueError:
                        pass

        if not valid_folders:
            return None

        # Sort the folders by their date and time components
        valid_folders.sort(key=lambda x: datetime.strptime(x.replace('-', ''), '%Y%m%d%H%M%S'), reverse=True)

        # Return the most recent folder
        return valid_folders[0]
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
