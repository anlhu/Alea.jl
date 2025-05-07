FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies for building Git from source
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    git

# Install Julia and create an alias
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz -O /usr/julia.tar.gz
RUN tar zxvf /usr/julia.tar.gz -C /usr
RUN echo 'alias julia="/usr/julia-1.11.5/bin/julia"' >> /etc/bash.bashrc
RUN echo 'alias juliap="julia --project"' >> /etc/bash.bashrc

# Install Sympy
RUN pip3 install sympy

# Install Julia dependencies
### Apparantly Julia wants to run as postCreateCommand instead of here
# RUN /usr/julia-1.11.5/bin/julia --project -e 'using Pkg; Pkg.instantiate()'

RUN git config --global --add safe.directory '*'
