FROM jupyter/tensorflow-notebook

USER $NB_USER

# Python packages from conda
RUN conda config --add channels conda-forge
RUN conda install --quiet --yes \
    h5py \
    graphviz &&\
    conda clean -tipsy && \
fix-permissions $CONDA_DIR

RUN mkdir Workspace
COPY ./Workspace Workspace

USER $NB_USER

