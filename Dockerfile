FROM jupyter/tensorflow-notebook

USER $NB_USER

# Python packages from conda
RUN conda config --add channels conda-forge
RUN conda install --quiet --yes \
    h5py \
    pandas \
    seaborn \
    graphviz &&\
    conda clean -tipsy && \
fix-permissions $CONDA_DIR

COPY ./Workspace Workspace

VOLUME /home/jovyan/Workspace