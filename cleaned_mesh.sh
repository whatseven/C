input_dir=$1
input_name=$2
output_dir=$3

echo "==============================================================";
echo "Cleaned mesh and split into different object";
echo "Need obj mesh (Z up) without ground floor";
echo "Check \"$1\" is the model directory, \"$2\" is the name of the mesh (under model directory) and \"$3\" is the place to output result";
echo "==============================================================";
echo "";
echo "";
echo "";

./build/src/rename_material/RelWithDebInfo/rename_material.exe --model_directory $input_dir --model_name $input_name --output_directory $output_dir

./build/src/split_obj_example/RelWithDebInfo/split_obj_example.exe --model_directory $output_dir --model_name renamed_material.obj --resolution .5