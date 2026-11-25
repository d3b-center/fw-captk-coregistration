# Python 3.12: the interpreter picsl_greedy's and SimpleITK's pinned wheels are tested with here
FROM python:3.12-slim-bookworm

# Setup default flywheel/v0 directory
ENV FLYWHEEL="/flywheel/v0"
WORKDIR ${FLYWHEEL}

# Install required libraries: Greedy and SimpleITK at the versions the registration was checked with, then the gear's
COPY requirements_greedy.txt requirements_main.txt $FLYWHEEL/
RUN pip install --no-cache-dir -r $FLYWHEEL/requirements_greedy.txt
RUN pip install --no-cache-dir -r $FLYWHEEL/requirements_main.txt
RUN pip check

# Copy main files into working directory
COPY run.py manifest.json ${FLYWHEEL}/
COPY gear_functions ${FLYWHEEL}/gear_functions

# Configure entrypoint
RUN chmod a+x $FLYWHEEL/run.py
ENTRYPOINT ["python","/flywheel/v0/run.py"]
