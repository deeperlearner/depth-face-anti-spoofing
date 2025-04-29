# This script run create depth of live face by using 3DDFA_V2
# bash depth_face.sh

cd 3DDFA_V2
# sh ./build.sh

oulu_base="/media/back/home/chuck/Dataset/Oulu_Npu/"
txt_path="oulu_filelist.txt"
oulu_path="${oulu_base}${txt_path}"
while IFS=' ' read -r file liveness _; do
    if [[ "$liveness" == "1" ]]; then
        echo "Filename: $file"
        echo "Liveness: $liveness"
        full_path="${oulu_base}crop/train/${file}"
        if [[ -f "$full_path" ]]; then
            echo "Exists: $full_path"
            python3 demo.py -f $full_path --onnx -o depth --output_path $(dirname $full_path)
        else
            echo "Missing: $full_path"
        fi
    fi
done < "$oulu_path"

# for f in /path/to/face.jpg
# do
#     python3 demo.py -f $f --onnx -o depth --output_path $(dirname $f)
# done
cd ..
