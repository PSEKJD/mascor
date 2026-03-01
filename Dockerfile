FROM nvcr.io/nvidia/pytorch:23.04-py3

# Default package
RUN apt-get update && apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Anaconda installation
ENV CONDA_DIR=/opt/conda
RUN curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# custom env named myenv & package installation
RUN conda create -n myenv python=3.9 -y \
    && conda run -n myenv pip install --upgrade pip \
    && conda run -n myenv pip install packaging==21.3 \
    && conda run -n myenv pip install transformers==4.5.1 \
    && conda run -n myenv pip install scikit-learn==1.5.0 \
    && conda run -n myenv pip install seaborn==0.13.2 \
    && conda run -n myenv pip install numpy matplotlib==3.9.2 pandas \
    && conda run -n myenv pip install -U "ray[rllib]==2.7.0" \
    && conda run -n myenv pip install numpy==1.26.3 \
    && conda run -n myenv pip install pyomo==6.7.3 \
    && conda run -n myenv pip install gurobipy \
    && conda run -n myenv pip install gymnasium==0.28.1 botorch==0.10.0 \
    && conda clean -afy

ENV PATH="/opt/conda/envs/myenv/bin:$CONDA_DIR/bin:$PATH"

# mounting gurobli liscens
RUN mkdir -p /root/.gurobi
COPY gurobi.lic /root/.gurobi/gurobi.lic
RUN chmod 600 /root/.gurobi/gurobi.lic
ENV GRB_LICENSE_FILE=/root/.gurobi/gurobi.lic

# working directory setting
WORKDIR /workspace

# Debugging 
RUN conda run -n myenv python --version && conda info && conda list -n myenv

ENTRYPOINT ["python", "-u"]
CMD ["excute_rbdo.py", "--scenario-size", "500", "--optim-iter", "20"]