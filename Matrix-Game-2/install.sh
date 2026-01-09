pip3 install setuptools
pip3 install -r requirements.txt 
pip install flash-attn --no-build-isolation
python setup.py develop

hf download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
