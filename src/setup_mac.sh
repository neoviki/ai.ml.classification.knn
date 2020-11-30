#Exit virtual environment
#deactivate

#setup environment for macos
rm -rf my_env
pip2 install virtualenv

#Create virtual environment
python2 -m virtualenv my_env

#Enter the virtual environment "my_env" and execute command
#bash -c "source my_env/bin/activate; pip2 install numpy scipy pandas scikit-learn matplotlib"

bash -c "source my_env/bin/activate; pip2 install -r ./requirement.txt"

