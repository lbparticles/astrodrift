```
          ###      ######## ########### #########   ########    
       ### ###   ###    ###    ###     ###    ### ###    ###    
     ###   ###  ###           ###     ###    ### ###    ###     
   ########### ##########    ###     #########  ###    ###      
  ###     ###        ###    ###     ###    ### ###    ###       
 ###     ### ###    ###    ###     ###    ### ###    ###        
###     ###  ########     ###     ###    ###  ########          
      #########  #########  ########### ########## ###########  
     ###    ### ###    ###     ###     ###            ###       
    ###    ### ###    ###     ###     ###            ###        
   ###    ### #########      ###     ########       ###         
  ###    ### ###    ###     ###     ###            ###          
 ###    ### ###    ###     ###     ###            ###           
#########  ###    ### ########### ###            ###           
```

`astrodrift` is a python library that provides numerical integrator for arbitrary potentials specialising in galatic dynamics simulations. The library provides both cpu compiled and gpu accelerated integration methods utilising a rust-based backend. The library focuses on large quantities of non-interacting test particle integrations particularly useful for tidal stream dynamics. The library also provides interpolation of moving potential increased performance.

# INSTALLATION

```python -m pip install astrodrift```

```uv add astrodrift```

# DEPENDENCIES

```
  
```



# CONTRIBUTION

To install from source 

```git clone https://github.com/lbparticles/astrodrift```

To use the gpu functions calls you must have an nvidia gpu and drivers installed on your system, download them from the [official website]( https://www.nvidia.com/en-us/drivers/) or use your os package manager.

Inside the containers/, there are two provided Dockerfile to build ubuntu22 and ubuntu24 versions of an apptainer that has the necessary nvidia toolkits installed. make sure that you have installed docker and apptainer installed. There are also example shell scripts that can be modified to create a docker container in the root of project -- WORKDIR. There is a translation script from docker to apptainer. There is a run.sh script that opens the apptainer with the project bound to /data/astrodrift and sets it to the current working directory.

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes
