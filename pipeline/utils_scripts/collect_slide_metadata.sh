# Functions utilized in this pipeline
function slidepath () {
	local path=$(realpath /mnt/*/OncoScope/tubeID_${slide:0:5}/*/slideID_${slide}/bzScanner/proc)
	path=$(echo $path | awk '{print $1}')
	echo $path
}
function slidesuffix () {
	local path=$(slidepath)
	suffix=$(ls $path/Tile000001.* | awk -F . '{print $NF}')
	suffix=$(echo $suffix | awk '{print $1}')
	echo $suffix
}

# create slides_suffix.txt file which includes slide ids and their image format
CURRENT_PATH=$(pwd)
cd $PIPELINE_PATH
input_file=$(cat config/config.yml | grep slides: | awk -F : '{print $2}')
output_file=$(cat config/config.yml | grep slides_metadata: | awk -F : '{print $2}')
echo "$input_file    $output_file"
slides=$(cat $input_file)
test -f $output_file && rm -rf $output_file
for slide in $slides
	do
		slide_path=$(slidepath)
		suffix=$(slidesuffix)
		echo -e "${slide}\t${suffix}\t${slide_path}" >> $output_file
	done
cd $CURRENT_PATH