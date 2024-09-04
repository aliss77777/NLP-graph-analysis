NOTE: this is a research script which features many diagnostics for exploration. Built as modular components to be cloned and remixed as needed.

Steps to run this project:

- Set-up: run the requirements.txt file to install the necessary libraries (to a virtual environment if desired)\


If you run all the notebooks in the same folder the pipeline should work, capturing the outputs of each step as the input to the next.

Step 1: Data Ingestion from Twitter (deprecated)
- Twitter API developer credentials needed
- exports a series of CSVs which are consumed in Step 2

Step 2: Adjacency List Creation
- consumes the files from Step 1
- performs text processing in TextBlob and spaCy
- run this from terminal before running spaCy to download language corpus: 
 -- python -m spacy download en_core_web_sm
- creates an adjacency list and exports it as a CSV
- update to the script (May 2022) performs community detection in python and exports a file in GraphML format for visualization in Gephi or other visualization tool for networks

Step 3: Summary Statistics
- This consumes the output of the adjacency list, merges with the original data files, and creates a number of summary statistics for explainability and actionability

questions: aliss77777@gmail.com

