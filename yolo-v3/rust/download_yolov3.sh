FIXTURE=https://github.com/apepkuss/openvino-ir-modelzoo/tree/main/yolo-v3-onnx/FP32/
TODIR=model

if [ ! -f $TODIR/yolo-v3-onnx.bin ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/yolo-v3-onnx.bin
fi
if [ ! -f $TODIR/yolo-v3-onnx.xml ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/yolo-v3-onnx.xml
fi
