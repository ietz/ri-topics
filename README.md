# ri-topics
## Setup
### With docker
1. Create a copy of `.env.example.docker` in the repository root directory and name it `.env`
1. Update the values in `.env` as needed
1. Build and run the docker image by executing
   ```bash
   docker build --rm -t ri-topics:latest .
   docker run --rm -it --env-file .env -p 8888:8888 ri-topics:latest
   ```

### Without docker
1. Install the [Anaconda python runtime](https://anaconda.org/)
1. Create the python environment with  
   `conda env create -f environment.yml`  
   or update an existing `ri-topics` environment to match remote dependency changes by executing  
   `conda env update -f environment.yml --prune`
1. Activate the conda environment  
   `conda activate ri-topics`
1. Create a copy of `.env.example.local` in the repository root directory and name it `.env`
1. Update the values in `.env` as needed
1. Run the service  
   `python main.py`

## Running tests
To generate the SonarQube `coverage-reports/coverage.xml` as well as the user friendly HTML report in `coverage-reports/html`, run
```bash
python -m pytest --cov=. --cov-report=xml --cov-report=html
```
