mkdir -p $HOME/dsml/data
mkdir -p $HOME/dsml/work
image=torch-cuda12:latest 
docker build -t $image .
jovyan_home=/home/jovyan
docker create --gpus all --name torch -P -v $HOME/dsml/work:$jovyan_home/work -v $HOME/dsml/data/:$jovyan_home/data torch-cuda12:latest start-notebook.sh --LabApp.token=''

