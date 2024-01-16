# This script contains functions for converting PyTorch model to ONNX, running ONNX models using ONNXRuntime, Quantizing and Performance Analysis of ONNX nodels

from datetime import datetime
from collections import deque
import gc
import itertools
import json
import math
import multiprocessing
import numpy as np
import os
import onnx
from PIL import Image
import random
import re
import shutil
import subprocess
import sys
import time
import timeit
import threading
import torch
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights
from torchvision import transforms
from typing import Optional, List

# global functions
def get_correct_type(input_type: str):
    if input_type == "(float16)":
        return np.float16
    
    elif input_type == "(float)":
        return np.float32
    
    elif input_type == "(double)":
        return np.double
    
    elif input_type == "(long)":
        return np.int_
    
    else:
            return
            
def get_random_input(input_shape: (tuple or list), input_type):  
    input_data = np.random.random(input_shape).astype(get_correct_type(re.search(r"\((.*)\)", input_type).group(0)))
     
    # img = Image.fromarray(np.reshape(np.squeeze(input_data, axis=0), (input_shape[2], input_shape[3], input_shape[1])), 'RGB')
    # img.save('test_input_data.png')
    
    return input_data

class ONNXInference:
    def __init__(self, model_name: str):
        self.model_name = str(model_name)
    
        self.workspace = os.path.join(os.path.expanduser('~'), 'IPU', 'onnxruntime')
        
        self.config_file_dir = os.path.join(self.workspace, 'config')
        self.xclbin_dir = os.path.join(self.workspace, 'config')
        self.cache_dir = os.path.join(self.workspace, '.cache')
        
        self.fp32_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'fp32', self.model_name)
        self.fp32_infer_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'fp32_infer', self.model_name)
        self.int8_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'int8', self.model_name)
        self.ryzen_ai_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'ryzen_ai', self.model_name)
        self.fp32_results_dir = os.path.join(self.workspace, 'results', 'fp32', self.model_name)
        self.int8_results_dir = os.path.join(self.workspace, 'results', 'int8', self.model_name)
        self.ryzen_ai_results_dir = os.path.join(self.workspace, 'results', 'ryzen_ai', self.model_name)
        self.fp32_prof_dir = os.path.join(self.workspace, 'profiling', 'fp32', self.model_name)
        self.int8_prof_dir = os.path.join(self.workspace, 'profiling', 'int8', self.model_name)
        self.ryzen_ai_prof_dir = os.path.join(self.workspace, 'profiling', 'ryzen_ai', self.model_name)
        self.fp32_agm_dir = os.path.join(self.workspace, 'agm_logs', 'fp32', self.model_name)
        self.int8_agm_dir = os.path.join(self.workspace, 'agm_logs', 'int8', self.model_name)
        self.ryzen_ai_agm_dir = os.path.join(self.workspace, 'agm_logs', 'ryzen_ai', self.model_name)
        
        for i in [self.cache_dir, self.fp32_onnx_dir, self.fp32_infer_onnx_dir, self.int8_onnx_dir, self.ryzen_ai_onnx_dir, self.fp32_results_dir, self.int8_results_dir, self.ryzen_ai_results_dir, \
                  self.fp32_prof_dir, self.int8_prof_dir, self.ryzen_ai_prof_dir, self.fp32_agm_dir, self.int8_agm_dir, self.ryzen_ai_agm_dir]:
            
            os.makedirs(i, exist_ok=True)
                
    def convert_torch_to_onnx(self,
                              model: torch.nn.Module,
                              pass_inputs: bool = False,
                              model_inputs: tuple[torch.Tensor] = None,
                              input_shape: (tuple or list) = None,
                              input_names: Optional[list] = None,
                              output_names: Optional[list] = None,
                              input_dynamic_axes: Optional[List[dict]] = None,
                              output_dynamic_axes: Optional[List[dict]] = None,
                              opset_version: int = 15,
                              use_dynamo: bool = False,
                              use_external_data: bool = False,
                              exists_ok: bool = True,
            ):
        
        if not use_dynamo:
            self.opset_version = opset_version
            self.use_external_data = use_external_data
        
        else:
            self.opset_version = 18
            self.use_external_data = False

        if not isinstance(model, torch.nn.Module):
            raise Exception("Model has to be of type torch.nn.Module")
        
        if os.path.exists(self.fp32_onnx_dir):
            if not exists_ok:
                shutil.rmtree(self.fp32_onnx_dir)
                os.mkdir(self.fp32_onnx_dir)
        
            else:
                print("\nONNX directory already exists. Skipping this step.")
                return
             
        # Export the model to ONNX
        if self.use_external_data:
            # onnx_model_path_p = os.path.join(self.fp32_onnx_dir, self.model_name + '_partial.onnx')
            onnx_model_path_p = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'llama2', 'model', 'partial', self.model_name + '_partial.onnx')

        # onnx_model_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')
        onnx_model_path = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'llama2', 'model', self.model_name + '.onnx')
        
        if pass_inputs and model_inputs is None:
            raise Exception("Input cannot be None")
        
        if not pass_inputs:
            if input_shape is None:
                raise Exception("Input shape cannot be None")
            
            else:
                model_inputs = tuple(torch.randn(*input_shape, requires_grad=False))

        with torch.no_grad():
            # set model to eval
            model.eval()
            
            print("\nConverting {} from PyTorch to ONNX...\n".format(self.model_name.capitalize()))
            # convert pytorch model to onnx
            if not use_dynamo:
                dynamic_axes = {}
        
                for i in range(len(input_dynamic_axes)):
                    dynamic_axes[str(input_names[i])] = input_dynamic_axes[i]
                
                for i in range(len(output_dynamic_axes)):
                    dynamic_axes[str(output_names[i])] = output_dynamic_axes[i]

                torch.onnx.export(
                    model,
                    model_inputs,
                    onnx_model_path_p if self.use_external_data else onnx_model_path,
                    export_params=True,
                    do_constant_folding=True,
                    opset_version=self.opset_version,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                )
            
            else:
                torch._dynamo.config.dynamic_shapes = True
                torch._dynamo.config.capture_scalar_outputs = True
                torch._dynamo.config.automatic_dynamic_shapes = True

                kwargs = {}

                for i in range(len(input_names)):
                    kwargs[str(input_names[i])] = model_inputs[i]

                export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
                          
                export_output = torch.onnx.dynamo_export(model, **kwargs, export_options=export_options)
                export_output.save(onnx_model_path)

        if self.use_external_data:
            print("\nSaving external data to one file...\n")

            # try freeing memory
            gc.collect()

            onnx_model = onnx.load(onnx_model_path_p, load_external_data=True)
            
            onnx.save_model(
                onnx_model,
                onnx_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=self.model_name + "-data",
                size_threshold=1024,
            )
        
        try:
            onnx.checker.check_model(onnx_model_path)

        except onnx.checker.ValidationError as e:
            raise Exception(e)

        print("\nSuccessfully converted PyTorch model to ONNX!!\n")
        
        return
        
    def quantize(self, shape_infer: bool = True, external_data_format: bool = False, pass_input: bool = False, data_directory: str = None, image_type: str = None):
        input_model_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')
        
        if shape_infer:
            infer_model_path = input_model_path.replace('fp32', 'fp32_infer')
            infer_model_path = infer_model_path[:-5] + '_infer.onnx'
        else:
            infer_model_path = input_model_path
            
        external_data_location = os.path.dirname(infer_model_path) if external_data_format else None
        
        if shape_infer:
            quantized_model_path = infer_model_path.replace('fp32_infer', 'ryzen_ai')
        else:
            quantized_model_path = infer_model_path.replace('fp32', 'ryzen_ai')
        
        quantized_model_path = quantized_model_path[:-5] + '_int8.onnx'

        from onnxruntime import InferenceSession
        from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader
        import vai_q_onnx
        
        class ImageCalibrationData(CalibrationDataReader):
            def __init__(self, onnx_model_path: str, model_name: str, data_directory: str, num_input_data: int = 250, image_type: str = None):
                self.image_type = image_type
                self.model_name = model_name
                
                dummy_session = InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
                self.input_names = [str(_input.name) for _input in dummy_session.get_inputs()]
                
                self.batch_size = 25
                
                if self.image_type == 'duts':
                    self.max_total_inputs = 50  
                else:
                    self.max_total_inputs = num_input_data
                
                # initial input data batch
                self.input_feed = iter(self.get_input_batch())

            def get_input_batch(self):
                if self.image_type == 'imagenet':
                    if 'resnet50' in self.model_name:
                        weights = ResNet50_Weights.DEFAULT
                    elif 'mobilenetv2' in self.model_name:
                        weights = MobileNet_V2_Weights.DEFAULT
                    else:
                        weights = None
                    
                    if weights is not None:
                        img_preprocess = weights.transforms()
                        
                    data_dirs = [os.path.join(data_directory, data_path) for data_path in os.listdir(data_directory)]
                    
                    data_paths = []
                    for data_dir in data_dirs:
                        partial_data_paths = os.listdir(data_dir)
                        data_paths.append(os.path.join(data_dir, partial_data_paths[random.randint(0, len(partial_data_paths) - 1)]))
                    
                    data = []
                    random.shuffle(data_paths)

                    for i in range(self.batch_size):
                        try:
                            img = img_preprocess(read_image(data_paths[i])).unsqueeze(0).numpy()
                            data.append(img)
                        
                        except Exception as _:
                            pass
                    
                    del data_dirs
                
                elif self.image_type == 'duts':
                    data_paths = [os.path.join(data_directory, data_dir) for data_dir in os.listdir(data_directory)]
                    random.shuffle(data_paths)
                    
                    data = [transforms.functional.convert_image_dtype(read_image(data_paths[i]), dtype=torch.float).unsqueeze(0).detach().numpy() for i in range(self.batch_size)]
                    
                else:
                    data_paths = [os.path.join(data_directory, data_path) for data_path in os.listdir(data_directory)]
                    random.shuffle(data_paths)
                    
                    data = [read_image(data_paths[i]).unsqueeze(0).detach().numpy() for i in range(self.batch_size)]
                
                self.max_total_inputs -= self.batch_size
                
                del data_paths
                
                input_feed = [{_input_name: _data for _input_name in self.input_names} for _data in data]
                
                del data
                gc.collect()

                return input_feed
                
            def get_next(self):
                next_input = next(self.input_feed, None)
                
                if next_input:
                    return next_input
                
                del self.input_feed
                gc.collect()
                
                if self.max_total_inputs >= self.batch_size:
                    self.input_feed = iter(self.get_input_batch())
                    
                    return next(self.input_feed, None)
                
                else:
                    return None 
        
        if shape_infer:
            from onnxruntime.quantization.shape_inference import quant_pre_process
            
            quant_pre_process(
                input_model_path,
                infer_model_path,
                skip_optimization=external_data_format,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                int_max=(2**31 - 1),
                verbose=1,
                save_as_external_data=external_data_format,
                all_tensors_to_one_file=external_data_format,
                external_data_location=external_data_location,
                external_data_size_threshold=1024,
            )
        
        print("\nQuantizing onnx model...\n")
        
        # Quantize with calibration
        vai_q_onnx.quantize_static(
            infer_model_path,
            quantized_model_path,
            calibration_data_reader=ImageCalibrationData(infer_model_path, self.model_name, data_directory=data_directory, image_type=image_type) if pass_input else None,
            quant_format=QuantFormat.QOperator,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            enable_dpu=True,
            extra_options={"ActivationSymmetric": True},
        )

        try:
            onnx.checker.check_model(quantized_model_path)
        
        except onnx.checker.ValidationError as e:
            raise Exception(e)
        
        print("\nSuccessfully quantized onnx model!!\n")
        
        return
    
    def start_processing_timed(self, latency_per_iteration: deque, ort_session, output_feed: list, input_feed: dict, runtime: int, profiling: bool):
        # ignore 1st run
        if not profiling:
            _time = timeit.timeit(lambda: ort_session.run(output_feed, input_feed), number=1)
        
        # run benchmark for %runtime% seconds
        latency_per_iteration.append(timeit.repeat(lambda: ort_session.run(output_feed, input_feed), number=1, repeat=math.ceil(runtime / _time)))
    
    def start_processing_iter(self, latency_per_iteration: deque, ort_session, output_feed: list, input_feed: dict, iterations: int, profiling: bool):
        # ignore 1st run
        if not profiling:
            _ = timeit.timeit(lambda: ort_session.run(output_feed, input_feed), number=1)
        
        # run benchmark for %runtime% seconds
        latency_per_iteration.append(timeit.repeat(lambda: ort_session.run(output_feed, input_feed), number=1, repeat=iterations))
                
    def _inference(self,
                   onnx_model_path: str,
                   config_file_path: str,
                   cache_dir: str,
                   pass_input: bool,
                   model_inputs: [torch.Tensor],
                   benchmark: bool,
                   profiling: bool,
                   runtime: int,
                   iterations: int,
                   inf_type: str,
                   num_threads: int,
                   verbosity: int,
                   timestamp: str,
            ):
                  
        import onnxruntime as ort
        
        latency_per_iteration = deque()
            
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.log_severity_level = verbosity
        
        if benchmark:
            sess_options.log_severity_level = 3
        
        if profiling:
            sess_options.enable_profiling = True
            sess_options.log_severity_level = verbosity

        # default: let onnxruntime choose
        # sess_options.intra_op_num_threads = 0
        # sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
         
        if inf_type == 'ryzen_ai':
            ort_session = ort.InferenceSession(
                onnx_model_path,
                providers=["VitisAIExecutionProvider"],
                sess_options=sess_options,
                provider_options=[
                    {
                        "config_file": config_file_path,
                        "cacheDir": cache_dir,
                        "cacheKey": self.model_name + '_ryzen_ai_' + timestamp,
                    },
                ],
            )
            
            if "VitisAIExecutionProvider" not in ort_session.get_providers():
                raise EnvironmentError("ONNXRuntime does not support VitisAIExecutionProvider. Build ONNXRuntime appropriately")
            
            # IPU compilation takes place when the session is created

        else:
            ort_session = ort.InferenceSession(
                onnx_model_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            )

        # Disable CPU fallback
        ort_session.disable_fallback()

        inputs = ort_session.get_inputs()

        outputs = ort_session.get_outputs()
        
        # Outputs
        output_feed = [output.name for output in outputs]
        
        if benchmark:
            inference_outputs = None
            
            # rng = np.random.default_rng()

            if pass_input:
                input_feed = {_input.name: model_inputs[i].detach().numpy() for _input in inputs}
            
            else:
                # input_feed[input_name] = rng.random(size=tuple(_input.shape), dtype=np.float32)
                input_feed = {_input.name: get_random_input([1] + _input.shape[1:], _input.type) for _input in inputs}

            # pool_thread = threading.Thread(target=self.start_processing_timed, args=(latency_per_iteration, ort_session, output_feed, input_feed, runtime, profiling))
            pool_thread = threading.Thread(target=self.start_processing_iter, args=(latency_per_iteration, ort_session, output_feed, input_feed, iterations, profiling))
               
            # start measuring time
            start = time.perf_counter()

            # wait a minimum time 
            while time.perf_counter() - start < 2:
                pass
            
            print("\nStarting benchmark for {} seconds...\n".format(runtime), flush=True)
            
            # start the thread
            # each thread creates a list of sessions of the same dimension of the data queue, then runs them.
            pool_thread.start()
            
            # wait for the thread to finish
            pool_thread.join()
            
            total_run_time = time.perf_counter() - start
            
            if profiling:
                prof_file = ort_session.end_profiling()
                print(prof_file, flush=True)
            
            print("\nSucessfully completed ONNXRuntime Benchmark!!\n", flush=True)
            
            return latency_per_iteration 

        else:
            if pass_input:
                input_feed = {_input.name: model_inputs[0].detach().numpy() for _input in inputs}
            
            else:
                input_feed = {_input.name: np.random.random(_input.shape).astype(self.get_correct_type(re.search(r"\((.*)\)", _input.type).group(0))) for _input in inputs} 
                    
            print("\nStarting ONNXRuntime Inference...\n")
            
            start = time.perf_counter()
            
            inference_outputs = ort_session.run(output_feed, input_feed)
            
            print(time.perf_counter() - start)
        
            print("\nSucessfully completed ONNXRuntime Inference!!\n")
        
            return inference_outputs
            
    def start_inference(self,
                        instance_count: int = 1,
                        pass_input: bool = False,
                        model_inputs: [torch.Tensor] = None,
                        benchmark: bool = False,
                        profiling: bool = False,
                        runtime: int = 60,
                        iterations: int = 1000,
                        num_threads: int = 8,
                        inf_type: str = 'fp32',
            ):
        
        if inf_type == 'ryzen_ai':
            onnx_model_path = os.path.join(self.ryzen_ai_onnx_dir, self.model_name + '_infer_int8.onnx')
            results_dir = self.ryzen_ai_results_dir
            prof_dir = self.ryzen_ai_prof_dir
        
        elif inf_type == 'int8':
            onnx_model_path = os.path.join(self.int8_onnx_dir, self.model_name + '_infer_int8.onnx')
            results_dir = self.int8_results_dir
            prof_dir = self.int8_prof_dir
        
        else:
            onnx_model_path = os.path.join(self.ryzen_ai_onnx_dir, self.model_name + '_infer_int8.onnx')
            results_dir = self.fp32_results_dir
            prof_dir = self.fp32_prof_dir
        
        if profiling:
            self.instance_count = 1
            self.iterations = 5
        else:
            self.instance_count = instance_count
            self.iterations = iterations
        
        # ryzen_ai := int8 on ryzen ai processor           
        if inf_type == 'ryzen_ai':
            os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_Nx4_Overlay"
            os.environ['XLNX_ENABLE_CACHE'] = '1'

            xclbin_path = os.path.join(self.xclbin_dir, '1x4.xclbin')
            os.environ['XLNX_VART_FIRMWARE'] = xclbin_path
            
            config_file_path = os.path.join(self.config_file_dir, 'vaip_config.json')
            cache_dir = self.cache_dir
        
        else:
            config_file_path = None
            cache_dir = None
        
        _datetime = datetime.now().strftime("%m%d%Y%H%M%S")

        if benchmark == False:
            output = self._inference(onnx_model_path, config_file_path, cache_dir, pass_input, model_inputs, benchmark, profiling, runtime, self.iterations, inf_type, num_threads, 3, _datetime)
        
        else:
            output = None
            
            print("ONNX Model Path: {}\n".format(onnx_model_path))

            total_cpu = multiprocessing.cpu_count()

            if total_cpu < self.instance_count:
                self.instance_count = total_cpu
            
            processes = []
            
            # agm_proc = self.startAGM(inf_type, _datetime)                
    
            with multiprocessing.Pool(processes=self.instance_count) as pool:
                _processes = [pool.apply_async(self._inference, args=(onnx_model_path, config_file_path, cache_dir, pass_input, model_inputs, benchmark, profiling, runtime, self.iterations, inf_type, num_threads, 3, _datetime)) for _ in range(self.instance_count)]
                results = [result.get() for result in _processes]
                
            # self.stopAGM(agm_proc)
            
            for i, _deque in enumerate(results):
                for result in _deque:
                    with open(os.path.join(results_dir, self.model_name + '_' + _datetime + '_' + str(i+1) + '.json'), 'w', encoding="utf-8") as f:
                        json.dump(self.get_latency_result(result), f, ensure_ascii=False, indent=4)
            
            # if profiling:
            #     for i, key in enumerate(prof_files):
            #         os.replace(os.path.join(os.getcwd(), prof_files[key]), os.path.join(prof_dir, prof_files[key][:-5] + '_' + str(i+1) + '.json'))
        
        return output
        
    def startAGM(self, inf_type: str, timestamp: str, dpm_level: int = 7):
        if inf_type == 'ryzen_ai':
            self.agm_dir = self.ryzen_ai_agm_dir
        elif inf_type == 'int8':
            self.agm_dir = self.int8_agm_dir
        else:
            self.agm_dir = self.fp32_agm_dir
        
        filename = os.path.join(self.agm_dir, 'agm_log_dpm' + str(dpm_level) + '_' + str(self.instance_count) + 'col_' + timestamp + '.csv')
        
        # command = '"C:\\Program Files\\AMD Graphics Manager\\AMDGraphicsManager.exe" -pmLogSetup=C:\\Users\\Administrator\\IPU\\onnxruntime\\config\\ipu_setup.cfg -pmperiod=50 -pmStopCheck -pmOutput=' + filename
        command = '"C:\\Program Files\\AMD Graphics Manager\\AMDGraphicsManager.exe" -unilogsetup=C:\\Users\\Administrator\\IPU\\onnxruntime\\config\\ipu_setup.cfg -unilogperiod=50 -unilogstopcheck -unilogoutput=' + filename
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return process

    def stopAGM(self, process):
        filename = os.path.join(self.agm_dir, "terminate.txt")
        process = subprocess.run("echo > " + filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            process.terminate()
        except AttributeError:
            pass
    
    # adapted from https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/benchmark_helper.py
    def get_latency_result(self, latency_list: list, batch_size: int = 1):
        latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
        latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
        throughput = batch_size * (1000.0 / latency_ms)

        return {
            "test_times": len(latency_list),
            "latency_variance": f"{latency_variance:.2f}",
            "latency_90_percentile": f"{np.percentile(latency_list, 90) * 1000.0:.2f}",
            "latency_95_percentile": f"{np.percentile(latency_list, 95) * 1000.0:.2f}",
            "latency_99_percentile": f"{np.percentile(latency_list, 99) * 1000.0:.2f}",
            "average_latency_ms": f"{latency_ms:.2f}",
            "IPS": f"{throughput:.2f}",
        }
