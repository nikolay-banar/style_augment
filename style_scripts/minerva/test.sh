CONTENT_PATH=/home/ubuntu/pycharmProjects/style_transfer/classification_of_crops/images/style/test
STYLE_PATH=/home/ubuntu/pycharmProjects/style_transfer/classification_of_crops/images/style/test
SAVE_ROOT=/home/ubuntu/pycharmProjects/style_transfer/classification_of_crops/images/minerva/art_test
INTERPOLATION_WEIGHTS='[0.0,0.2,0.4,0.6,0.8,1.0]'
for INST in "Harp (3285)" "Lute (3394)" "Violin (3573)"; do
  i=0
  STYLE_FILES=(${STYLE_PATH}/"${INST}"/*)
  CONTENT_FILES=(${CONTENT_PATH}/"${INST}"/*)
  SAVE_PATH="${SAVE_ROOT}"/art0/"${INST}"

  if [ ! -e "${SAVE_PATH}" ]; then
      mkdir -p "${SAVE_PATH}"
  fi

  INDEX=()

  for f in "${STYLE_FILES[@]}"; do
  INDEX+=($i)
  i=$(( i + 1 ))
  done

  j=0
  INDEX=( $(shuf -e "${INDEX[@]}") )

  for f in "${STYLE_FILES[@]}"; do
  CUDA_VISIBLE_DEVICES=1 arbitrary_image_stylization_with_weights \
  --checkpoint=./style_models/arbitrary_style_transfer/model.ckpt \
  --output_dir="${SAVE_PATH}" \
  --style_images_paths="${f}" \
  --content_images_paths="${CONTENT_FILES[${INDEX[j]}]}" \
  --image_size=448 \
  --content_square_crop=False \
  --style_image_size=448 \
  --style_square_crop=False \
  --interpolation_weights=$INTERPOLATION_WEIGHTS \
  --logtostderr
    j=$(( j + 1 ))

  done
  find "${SAVE_PATH}" -type f ! -name '*_stylized_*' -delete
done

python /home/ubuntu/pycharmProjects/style_transfer/classification_of_crops/preprocess/square_images.py -path "${SAVE_ROOT}"/art0/


x=1
while [ $x -le 5 ]
do
  echo "Welcome $x times"
  y=$(( 2*$x))
  cp -r "${SAVE_ROOT}"/art0/ "${SAVE_ROOT}"/art${y}/ ;  find "${SAVE_ROOT}"/art${y}/ -type f ! -name "*_stylized_*_${x}.*" -delete
  x=$(( $x + 1 ))
done

find "${SAVE_ROOT}"/art0/ -type f ! -name "*_stylized_*_0.*" -delete