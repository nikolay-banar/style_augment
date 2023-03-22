CONTENT_PATH=./classification_of_crops/images/mimo/dev
STYLE_PATH=./classification_of_crops/images/minerva/dev

for INST in "Harp (3285)" "Lute (3394)" "Violin (3573)"; do
  i=0
  STYLE_FILES=(${STYLE_PATH}/"${INST}"/*)
  CONTENT_FILES=(${CONTENT_PATH}/"${INST}"/*)
  SAVE_PATH=./classification_of_crops/images/outstyle/dev/"${INST}"

  if [ ! -e "${SAVE_PATH}" ]; then
      mkdir -p "${SAVE_PATH}"
  fi

  for f in "${STYLE_FILES[@]}"; do

    CUDA_VISIBLE_DEVICES=1 arbitrary_image_stylization_with_weights \
  --checkpoint=./style_models/arbitrary_style_transfer/model.ckpt \
  --output_dir="${SAVE_PATH}" \
  --style_images_paths="${f}" \
  --content_images_paths="${CONTENT_FILES[i]}" \
  --image_size=224 \
  --content_square_crop=False \
  --style_image_size=224 \
  --style_square_crop=False \
  --logtostderr
    i=$(( i + 1 ))
#    echo index $i
#    echo total $total
#    echo "- Processing file: $f"
  done

done
