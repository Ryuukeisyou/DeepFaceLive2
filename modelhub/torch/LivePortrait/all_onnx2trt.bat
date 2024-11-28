cd faster_live_portrait
SET PYTHONEXECUTABLE=../../../../_internal/python/python.exe

:: warping+spade model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/warping_spade-fix.onnx
:: landmark model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/landmark.onnx
:: motion_extractor model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/motion_extractor.onnx -p fp32
:: face_analysis model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/retinaface_det_static.onnx
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx
:: appearance_extractor model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx
:: stitching model
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching.onnx
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_eye.onnx
"%PYTHONEXECUTABLE%" scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_lip.onnx
