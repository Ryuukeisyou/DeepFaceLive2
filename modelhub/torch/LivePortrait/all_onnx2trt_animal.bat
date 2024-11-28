cd faster_live_portrait
SET PYTHONEXECUTABLE=../../../../_internal/python/python.exe

:: warping+spade model
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/warping_spade-fix.onnx
:: motion_extractor model
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/motion_extractor.onnx -p fp32
:: appearance_extractor model
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/appearance_feature_extractor.onnx
:: stitching model
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching.onnx
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_eye.onnx
"%PYTHONEXECUTABLE%" ./scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_lip.onnx