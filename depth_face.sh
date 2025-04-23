# This script run create depth of live face by using 3DDFA_V2
# bash depth_face.sh

cd 3DDFA_V2
sh ./build.sh
for f in /path/to/face.jpg
do
    python3 demo.py -f $f --onnx -o depth --output_path $(dirname $f)
done
cd ..
