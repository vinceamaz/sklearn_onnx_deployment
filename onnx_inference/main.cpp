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
    int batch_size = 2;

    // onnxruntime 模型路径需要是宽字符 wstring
    std::string model_path = "model/pipeline_xgb.onnx";
    std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());

    // Read multiple input arrays
    std::vector<std::vector<float>> input_arrs;
    for (int i = 0; i < batch_size; ++i) {
        std::string arr_path = "input/array_data.bin";
        std::vector<float> input_arr = readArrayFromFile(arr_path);
        input_arrs.push_back(input_arr);
    }

    // ----------------------------------------------------------- 
    // 2. onnxruntime 初始化

    Ort::SessionOptions session_options;
    // 设置 logging level 为 ERROR
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "sklearn-onnx");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device: " << device_id << std::endl;
    auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
    // auto status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, device_id);
    Ort::Session session(env, w_model_path.c_str(), session_options);

    // ----------------------------------------------------------- 
    // 3. 从模型中读取输入和输入信息

    // print (name/shape) of inputs
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }

    // print (name/shape) of outputs
    std::vector<std::string> output_names;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    }

    // Assume model has 1 input node and 2 output nodes.
    assert(input_names.size() == 1 && output_names.size() == 2);

    // ----------------------------------------------------------- 
    // 4. 构造输入 tensor

    auto input_node_dims = input_shapes;
    assert(input_node_dims[0] == -1);  // symbolic dimensions are represented by a -1 value
    input_node_dims[0] = batch_size;
    auto total_number_elements = calculate_product(input_node_dims);

    std::vector<float> input_tensor_values(total_number_elements);
    // 将 input_arrs 复制到 input_tensor_values
    for (const auto& input_arr : input_arrs) {
        input_tensor_values.insert(input_tensor_values.end(), input_arr.begin(), input_arr.end());
    }

    // 将读取的 vector 转成 ort tensor
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_node_dims));

    // double-check the dimensions of the input tensor
    assert(input_tensors[0].IsTensor() &&
        input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_node_dims);
    std::cout << "input_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    // pass data through model
    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
        [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
        [&](const std::string& str) { return str.c_str(); });

    // ----------------------------------------------------------- 
    // 5. 推理

    std::cout << "Running model..." << std::endl;

    try {

        auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
            input_names_char.data(),
            input_tensors.data(),
            input_names_char.size(),
            output_names_char.data(),
            output_names_char.size());

        std::cout << "Done inference!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

        // 遍历读取每一个输出节点, 可以处理 tensor 和 sequence 的情况
        for (std::size_t i = 0; i < output_names.size(); ++i) {
            // GetONNXType(): [UNKNOWN, TENSOR, SEQUENCE, MAP, OPAQUE, SPARSETENSOR, OPTIONAL]
            if (output_tensors[i].GetTypeInfo().GetONNXType() == 1) {
                // 输出是 tensor 的情况
                const float* output_data_ptr = output_tensors[i].GetTensorMutableData<float>();
                
                for (int j = 0; j < batch_size; ++j) {

                    const float* out = output_data_ptr + j;
                    std::cout << j + 1 << "'th instance in batch (output name: " << output_names[i] << "): " << *out << std::endl;
                }
            }
            else if (output_tensors[i].GetTypeInfo().GetONNXType() == 2) {
                // 输出是 sequence 的情况

                for (size_t j = 0; j < batch_size; ++j) {

                    Ort::Value seq_out = output_tensors[i].GetValue(static_cast<int>(j), allocator);

                    // 遍历 sequence 里面每一个元素
                    for (size_t val_idx = 0; val_idx < seq_out.GetCount(); ++val_idx) {
                        // ONNXTensorElementDataType, string 格式的读取会不一样
                        // https://github.com/microsoft/onnxruntime/blob/9e67b88c8312e124c3127c6ee0833c110a18fd5a/onnxruntime/test/shared_lib/test_nontensor_types.cc#L97
                        auto tensor_dtype = seq_out.GetValue(static_cast<int>(val_idx), allocator).GetTensorTypeAndShapeInfo().GetElementType();
                        if (tensor_dtype == 1) {
                            // float
                            Ort::Value float_out = seq_out.GetValue(static_cast<int>(val_idx), allocator);
                            const float* float_result_ptr = float_out.GetTensorMutableData<float>();
                            std::cout << j + 1 << "'th instance in batch (output name: " << output_names[i] << ") float output: " << *float_result_ptr << std::endl;
                        }
                        else if (tensor_dtype > 1 && tensor_dtype < 8) {
                            // int
                            Ort::Value int_out = seq_out.GetValue(static_cast<int>(val_idx), allocator);
                            const int* int_result_ptr = int_out.GetTensorMutableData<int>();
                            std::cout << j + 1 << "'th instance in batch (output name: " << output_names[i] << ") int output: " << *int_result_ptr << std::endl;
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
