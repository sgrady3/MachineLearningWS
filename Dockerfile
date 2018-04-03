FROM jupyter/tensorflow-notebook

USER $NB_USER

# Python packages from conda
RUN conda install --quiet --yes \
    -c conda-forge python-graphviz &&\
    conda clean -tipsy && \
fix-permissions $CONDA_DIR

RUN mkdir Workspace
COPY ./Workspace Workspace

USER $NB_USER

