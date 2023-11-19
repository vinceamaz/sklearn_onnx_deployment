#include "main.h"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"  // cpu provider
#include <fstream>
#include <assert.h>
#include <sstream>
#include <algorithm>  // std::generate

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

// 用于计算 vector 元素个数
int calculate_product(const std::vector<std::int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

std::vector<float> readArrayFromFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<float> array_data;

    if (file.is_open()) {
        // Read the file into a vector of doubles (assuming the array contains doubles)
        array_data = std::vector<float>(std::istreambuf_iterator<char>(file), {});

        file.close();
    }
    else {
        std::cout << "File not found or unable to open." << std::endl;
    }

    return array_data;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}


// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/model-explorer.cpp
int main()
{
    int device_id = 0;
    // onnxruntime 模型路径需要是宽字符 wstring
    std::string model_path = "model/pipeline_xgb.onnx";
    std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());

    // 读取输入数据
    std::string arr_path = "input/array_data.bin";
    std::vector<float> input_tensor_values = readArrayFromFile(arr_path);

    // ----------------------------------------------------------- 
    // 2. onnxruntime 初始化

    Ort::SessionOptions session_options;
    // 设置 logging level 为 ERROR
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "sklearn-onnx");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device: " << device_id << std::endl;
    auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
    // auto status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, device_id);
    Ort::Session session_(env, w_model_path.c_str(), session_options);

    // ----------------------------------------------------------- 
    // 3. 从模型中读取输入和输入信息

    // print (name/shape) of inputs
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
        input_names.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
        input_shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto& s : input_shapes) {
        if (s < 0) {
            s = 1;
        }
    }

    // print (name/shape) of outputs
    std::vector<std::string> output_names;
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
        output_names.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
    }

    // Assume model has 1 input node and 2 output nodes.
    assert(input_names.size() == 1 && output_names.size() == 2);

    // ----------------------------------------------------------- 
    // 4. 构造输入 tensor

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes;
    auto total_number_elements = calculate_product(input_shape);

    // generate random numbers in the range [0, 255]
    // std::vector<float> input_tensor_values(total_number_elements);
    // std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });

    // 将读取的 vector 转成 ort tensor
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

    // 确保输入 ort tensor 的 shape 与 onnx 模型要求的一样
    assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    std::cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    // pass data through model
    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
        [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
        [&](const std::string& str) { return str.c_str(); });


    std::cout << "Running model..." << std::endl;
    try {
        auto output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names_char.data(), input_tensors.data(),
            input_names_char.size(), output_names_char.data(), output_names_char.size());
        std::cout << "Done!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

        // 遍历读取每一个输出节点, 可以处理 tensor 和 sequence 的情况
        for (std::size_t i = 0; i < output_names.size(); i++) {
            // GetONNXType(): [UNKNOWN, TENSOR, SEQUENCE, MAP, OPAQUE, SPARSETENSOR, OPTIONAL]
            if (output_tensors[i].GetTypeInfo().GetONNXType() == 1) {
                // 输出是 tensor 的情况
                const float* output_data_ptr = output_tensors[i].GetTensorMutableData<float>();
                std::cout << output_names[i] << " is " << *output_data_ptr << std::endl;
            }
            else if (output_tensors[i].GetTypeInfo().GetONNXType() == 2) {
                // 输出是 sequence 的情况
                size_t num_values = output_tensors[i].GetCount();

                for (size_t idx = 0; idx < num_values; ++idx) {

                    Ort::Value seq_out = output_tensors[i].GetValue(static_cast<int>(idx), allocator);

                    // 遍历 sequence 里面每一个元素
                    for (size_t val_idx = 0; val_idx < seq_out.GetCount(); ++val_idx) {
                        // ONNXTensorElementDataType, string 格式的读取会不一样
                        // https://github.com/microsoft/onnxruntime/blob/9e67b88c8312e124c3127c6ee0833c110a18fd5a/onnxruntime/test/shared_lib/test_nontensor_types.cc#L97
                        auto tensor_dtype = seq_out.GetValue(static_cast<int>(val_idx), allocator).GetTensorTypeAndShapeInfo().GetElementType();
                        if (tensor_dtype == 1) {
                            // float
                            Ort::Value float_out = seq_out.GetValue(static_cast<int>(val_idx), allocator);
                            const float* float_result_ptr = float_out.GetTensorMutableData<float>();
                            std::cout << output_names[i] << "'s float output is " << *float_result_ptr << std::endl;
                        }
                        else if (tensor_dtype > 1 && tensor_dtype < 8) {
                            // int
                            Ort::Value int_out = seq_out.GetValue(static_cast<int>(val_idx), allocator);
                            const int* int_result_ptr = int_out.GetTensorMutableData<int>();
                            std::cout << output_names[i] << "'s int output is " << *int_result_ptr << std::endl;
                        }
                        else {
                            std::cout << "Encountered new ONNXTensorElementDataType type: " << tensor_dtype << std::endl;
                        }
               
                    }
                }
            }
        }
        
    }
    catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }

    

    
}
