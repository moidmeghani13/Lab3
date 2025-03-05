#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>
#include <map>

#include "file_utils.h"
#include "stream_utils.h"
#include "array4d.h"

using namespace std;

template <class T>
class Network {
public:
    Network(std::string cfg_file_name);
    ~Network();

    int obtain_parameters();
    int conv_convert(int layer_id, int padding, int stride, Array3D<T>& initial_input, Array4D<T>& initial_kernel,
             Array2D<T>& input_matrix, Array2D<T>& kernel_matrix);
    int conv_convert_stream(int layer_id, int padding, int stride, Stream<T>& input, Stream<T>& output);

    void initialize();
    std::string get_parameters();

    const std::vector<int> &getInput_height() const;
    void setInput_height(const std::vector<int>& input_height);
    const std::vector<int> &getInput_width() const;
    void setInput_width(const std::vector<int>& input_width);
    const std::vector<int> &getInput_channel() const;
    void setInput_channel(const std::vector<int>& input_channel);

    const std::vector<int> &getKernel_dimension() const;
    void setKernel_dimension(const std::vector<int>& kernel_dimension);
    const std::vector<int> &getKernel_size() const;
    void setKernel_size(const std::vector<int> &kernel_size);
    const std::vector<int> &getKernel_channel() const;
    void setKernel_channel(const std::vector<int> &kernel_channel);

    const std::vector<int> &getOutput_height() const;
    void setOutput_height(const std::vector<int> &output_height);
    const std::vector<int> &getOutput_width() const;
    void setOutput_width(const std::vector<int> &output_width);
    const std::vector<int> &getOutput_channel() const;
    void setOutput_channel(const std::vector<int> &output_channel);

    int getLayer_number() const;
    void setLayer_number(int layer_number);

private:
    int layer_number;

    std::vector<int> input_height;
    std::vector<int> input_width;
    std::vector<int> input_channel;

    std::vector<int> kernel_dimension;
    std::vector<int> kernel_size;
    std::vector<int> kernel_channel;

    std::vector<int> output_height;
    std::vector<int> output_width;
    std::vector<int> output_channel;

    std::string cfg_file_name;
    File_utils<T> *cfg_util;
    std::vector<std::string> network_cfg_description;
};

template <class T>
Network<T>::Network(std::string cfg_file_name) {
    this->cfg_file_name = cfg_file_name;
}

template <class T>
Network<T>::~Network() {
    input_height.clear();
    input_width.clear();
    input_channel.clear();

    kernel_dimension.clear();
    kernel_size.clear();
    kernel_channel.clear();

    output_height.clear();
    output_width.clear();
    output_channel.clear();

    delete(cfg_util);
}

template <class T>
void Network<T>::initialize() {
    cfg_util = new File_utils<T>(cfg_file_name);
    cfg_util->parse_file();
}

template<class T>
std::string Network<T>::get_parameters() {
    std::string parameters = std::to_string(layer_number);
    parameters += "\n";
    for (int i = 0; i < layer_number; i++) {
        parameters += std::to_string(input_height[i]);
        parameters += " ";
        parameters += std::to_string(input_width[i]);
        parameters += " ";
        parameters += std::to_string(input_channel[i]);
        parameters += " ";

        parameters += std::to_string(kernel_dimension[i]);
        parameters += " ";
        parameters += std::to_string(kernel_size[i]);
        parameters += " ";
        parameters += std::to_string(kernel_channel[i]);
        parameters += " ";

        parameters += std::to_string(output_height[i]);
        parameters += " ";
        parameters += std::to_string(output_width[i]);
        parameters += " ";
        parameters += std::to_string(output_channel[i]);
        parameters += " ";
        parameters += "\n";
    }
    return parameters;
}


template<class T>
const std::vector<int> &Network<T>::getInput_height() const {
    return input_height;
}

template<class T>
void Network<T>::setInput_height(const std::vector<int> &input_height) {
    Network::input_height = input_height;
}

template<class T>
const std::vector<int> &Network<T>::getInput_width() const {
    return input_width;
}

template<class T>
void Network<T>::setInput_width(const std::vector<int> &input_width) {
    Network::input_width = input_width;
}

template<class T>
const std::vector<int> &Network<T>::getInput_channel() const {
    return input_channel;
}

template<class T>
void Network<T>::setInput_channel(const std::vector<int> &input_channel) {
    Network::input_channel = input_channel;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_dimension() const {
    return kernel_dimension;
}

template<class T>
void Network<T>::setKernel_dimension(const std::vector<int> &kernel_dimension) {
    Network::kernel_dimension = kernel_dimension;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_size() const {
    return kernel_size;
}

template<class T>
void Network<T>::setKernel_size(const std::vector<int> &kernel_size) {
    Network::kernel_size = kernel_size;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_channel() const {
    return kernel_channel;
}

template<class T>
void Network<T>::setKernel_channel(const std::vector<int> &kernel_channel) {
    Network::kernel_channel = kernel_channel;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_height() const {
    return output_height;
}

template<class T>
void Network<T>::setOutput_height(const std::vector<int> &output_height) {
    Network::output_height = output_height;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_width() const {
    return output_width;
}

template<class T>
void Network<T>::setOutput_width(const std::vector<int> &output_width) {
    Network::output_width = output_width;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_channel() const {
    return output_channel;
}

template<class T>
void Network<T>::setOutput_channel(const std::vector<int> &output_channel) {
    Network::output_channel = output_channel;
}

template<class T>
int Network<T>::getLayer_number() const {
    return layer_number;
}

template<class T>
void Network<T>::setLayer_number(int layer_number) {
    Network::layer_number = layer_number;
}

/***************************************************************/
/* Do not modify the above code.
   You are allowed to use the following global variables in your
   code. These are defined above.

   Begin your code here 	  			       */
/***************************************************************/

template <class T>
int Network<T>::obtain_parameters() {
    network_cfg_description = cfg_util->getFile_contents();

    layer_number = 0;

    input_height.clear();
    input_width.clear();
    input_channel.clear();

    kernel_dimension.clear();
    kernel_size.clear();
    kernel_channel.clear();

    output_height.clear();
    output_width.clear();
    output_channel.clear();
    /* Part I */
    /* Write your code here */

    int net_height = 0;
    int net_width = 0;
    int net_channel = 0;
    
    
    for (int i = 0; i < network_cfg_description.size(); i++){
        string line = network_cfg_description[i];
        
        if(line == "[net]"){
            string height = network_cfg_description[i+1].substr(network_cfg_description[i+1].find("=")+1);
            string width = network_cfg_description[i+2].substr(network_cfg_description[i+2].find("=")+1);
            string channel = network_cfg_description[i+3].substr(network_cfg_description[i+3].find("=")+1);
            net_height = stoi(height);
            net_width = stoi(width);
            net_channel = stoi(channel);

            // cout << network_cfg_description[i+1] << endl;
            // cout << network_cfg_description[i+2] << endl;
            // cout << network_cfg_description[i+3] << endl;
            
            // cout << "height: " << height << endl;
            // cout << "width: " << width << endl;
            // cout << "channel: " << channel << endl;
        }
        else if(line == "[convolutional]"){
            layer_number++;

            input_height.push_back(net_height);
            input_width.push_back(net_width);
            input_channel.push_back(net_channel);

            string batch_normalize = network_cfg_description[i+1].substr(network_cfg_description[i+1].find("=")+1);
            string filters = network_cfg_description[i+2].substr(network_cfg_description[i+2].find("=")+1);
            string k_size = network_cfg_description[i+3].substr(network_cfg_description[i+3].find("=")+1);
            string stride = network_cfg_description[i+4].substr(network_cfg_description[i+4].find("=")+1);
            string pad = network_cfg_description[i+5].substr(network_cfg_description[i+5].find("=")+1);
            string activation = network_cfg_description[i+6].substr(network_cfg_description[i+6].find("=")+1);

            // cout << "batch_normalize: " << batch_normalize << endl;
            // cout << "filters: " << filters << endl;
            // cout << "k_size: " << k_size << endl;
            // cout << "stride: " << stride << endl;
            // cout << "pad: " << pad << endl;
            // cout << "activation: " << activation << endl;

            kernel_dimension.push_back(stoi(filters));
            kernel_size.push_back(stoi(k_size));
            kernel_channel.push_back(net_channel);

            net_height = (net_height + 2*stoi(pad) - stoi(k_size))/stoi(stride) + 1;
            net_width = (net_width + 2*stoi(pad) - stoi(k_size))/stoi(stride) + 1;
            net_channel = stoi(filters);

            output_height.push_back(net_height);
            output_width.push_back(net_width);
            output_channel.push_back(net_channel);
        }
        else if(line == "[maxpool]"){
            string size = network_cfg_description[i+1].substr(network_cfg_description[i+1].find("=")+1);
            string stride = network_cfg_description[i+2].substr(network_cfg_description[i+2].find("=")+1);
            int size_val = stoi(size);
            int stride_val = stoi(stride);

            // cout << "size: " << size << endl;
            // cout << "stride: " << stride << endl;

            net_height = (net_height - size_val)/stride_val + 1;
            net_width = (net_width - size_val)/stride_val + 1;
        } 
        else {
            continue;
        }
    }
    
    return 0;
}

template <class T>
int Network<T>::conv_convert(int layer_id, int padding, int stride, Array3D<T>& initial_input, Array4D<T>& initial_kernel,
                              Array2D<T>& input_matrix, Array2D<T>& kernel_matrix) {
    /* Part II */
    /*Write your code here*/
    int numChannels=initial_input.Size_1d();
    int numHeight=initial_input.Size_3d();
    int numWidth=initial_input.Size_2d();

    int kernelSize=initial_kernel.Size_2d();
    int kernelMinHeight=0;
    int kernelMaxHeight=kernelMinHeight+kernelSize-1;
    int kernelMinWidth=0;
    int kernelMaxWidth=kernelMinWidth+kernelSize-1;

    int totalHeight = numHeight+2*padding;
    int totalWidth = numWidth+2*padding;
    
    Array2D<T> tempInputMatrix((((totalHeight-kernelSize)/stride)+1) * (((totalWidth-kernelSize)/stride)+1),kernelSize*kernelSize*numChannels);
    int currTempInputMatrixHeight = 0;

    for(int i=0;i<numChannels;i++){     //Padding
        Array2D<T> temp(numHeight+2*padding,numWidth+2*padding);
        for(int j=0;j<numHeight+2*padding;j++){ //height
            for(int k=0;k<numWidth+2*padding;k++){
                if(j<padding || j>=numHeight+padding || k<padding || k>=numWidth+padding){
                    temp[j][k]=0;
                }
                else{
                    temp[j][k]=initial_input[j-padding][k-padding][i];
                }
            }
        }

        currTempInputMatrixHeight = 0;
        for(int j=0;j<totalHeight-kernelSize+1;j+=stride){
            kernelMinHeight=j;
            kernelMaxHeight=j+kernelSize-1;
            for(int k=0;k<totalWidth-kernelSize+1;k+=stride){
                int kernelIndex = 0;
                kernelMinWidth=k;
                kernelMaxWidth=k+kernelSize-1;
                for(int l=kernelMinHeight;l<=kernelMaxHeight;l++){
                    for(int m=kernelMinWidth;m<=kernelMaxWidth;m++){
                        tempInputMatrix[currTempInputMatrixHeight][(kernelIndex*numChannels)+i] = temp[l][m];
                        kernelIndex++;
                    }
                }
                currTempInputMatrixHeight++;
            }
        }
        kernelMinHeight=0;
        kernelMaxHeight=kernelMinHeight+kernelSize-1;
        kernelMinWidth=0;
        kernelMaxWidth=kernelMinWidth+kernelSize-1;
    }

    input_matrix.resize(tempInputMatrix.Size_2d(),tempInputMatrix.Size_1d());
    for(int j=0;j<input_matrix.Size_2d();j++){
        for(int k=0;k<input_matrix.Size_1d();k++){
            input_matrix[j][k]=tempInputMatrix[j][k];
        }
    }


    Array2D<T> kernel_matrix_temp(initial_kernel.Size_1d()*initial_kernel.Size_2d()*initial_kernel.Size_3d(),initial_kernel.Size_4d());
    int kernel_row = 0;
    int kernel_col = 0;

    //Kernel Matrix
    for(int i=0;i<initial_kernel.Size_3d();i++){
        for(int j=0;j<initial_kernel.Size_2d();j++){
            for(int k=0;k<initial_kernel.Size_1d();k++){
                for(int l=0;l<initial_kernel.Size_4d();l++){
                    kernel_matrix_temp[kernel_row][kernel_col] = initial_kernel[l][i][j][k];
                    //cout << "kernel_matrix_temp[" << kernel_row << "][" << kernel_col << "] = " << kernel_matrix_temp[kernel_row][kernel_col] << endl;
                    //cout << "initial_kernel[" << l << "][" << i << "][" << j << "][" << k << "] = " << initial_kernel[l][i][j][k] << endl;
                    kernel_col++;
                }
                kernel_col=0;
                kernel_row++;
            }
        }
    }
    kernel_matrix.resize(kernel_matrix_temp.Size_2d(),kernel_matrix_temp.Size_1d());
    for(int j=0;j<kernel_matrix.Size_2d();j++){
        for(int k=0;k<kernel_matrix.Size_1d();k++){
            kernel_matrix[j][k]=kernel_matrix_temp[j][k];
        }
    }
    cout << endl;
    return 0;
}


template <class T>
int Network<T>::conv_convert_stream(int layer_id, int padding, int stride, Stream<T> &input, Stream<T> &output) {
    int buffer_size = (input_width[layer_id] + padding * 2) * input_channel[layer_id] * kernel_size[layer_id];
    T buffer[buffer_size];
    /* Part III */
    /*Write your code here*/
    int numElementsinBuffer = 0;
    int BufferStartingRow = 0;

    int maxNumRows = (2*padding+input_width[layer_id]);
    int currRow = 0;
    int maxNumCols = (2*padding+input_width[layer_id]);
    int currCol = 0;

    //initially fill the buffer with zeros
    for(int i=0; i<(2*padding+input_width[layer_id])*input_channel[layer_id]; i++){
        buffer[numElementsinBuffer] = 0;
        numElementsinBuffer++;
    }
    while(numElementsinBuffer < buffer_size){
        for(int i=0; i<input_channel[layer_id]; i++){
            buffer[numElementsinBuffer] = 0;
            numElementsinBuffer++;
        }
        for(int j=0; j<input_width[layer_id]*input_channel[layer_id]; j++){
            buffer[numElementsinBuffer] = input.read();
            numElementsinBuffer++;
        }
        for(int i=0; i<input_channel[layer_id]; i++){
            buffer[numElementsinBuffer] = 0;
            numElementsinBuffer++;
        }
    }

    for(int kRow = 0; kRow<maxNumRows; kRow+=stride){
        if(input.empty()){
            break;
        }
        for(int kCol = 0; kCol<maxNumCols; kCol+=stride){
            for(int i=kRow-BufferStartingRow; i<kRow+kernel_size[layer_id];kRow++){
                for(int j=kCol; j<kCol+kernel_size[layer_id];kCol++){
                    int bufIndex = input_channel[layer_id]*maxNumCols*i + input_channel[layer_id]*j;
                    for(int k=0; k<input_channel[layer_id]; k++){
                        output.write(buffer[bufIndex+k]);
                    }
                }
            }
        }
        numElementsinBuffer = 0;
        BufferStartingRow += stride;
        if(stride==kernel_size[layer_id]){
            while(numElementsinBuffer < buffer_size){
                for(int i=0; i<input_channel[layer_id]; i++){
                    buffer[numElementsinBuffer] = 0;
                    numElementsinBuffer++;
                }
                for(int j=0; j<input_width[layer_id]*input_channel[layer_id]; j++){
                    buffer[numElementsinBuffer] = input.read();
                    numElementsinBuffer++;
                }
                for(int i=0; i<input_channel[layer_id]; i++){
                    buffer[numElementsinBuffer] = 0;
                    numElementsinBuffer++;
                }
            }
        }
        else{
            for(int i = stride*(2*padding+input_width[layer_id])*input_channel[layer_id]; i<buffer_size; i++){
                buffer[numElementsinBuffer] = buffer[i];
                numElementsinBuffer++;
            }
            while(numElementsinBuffer < buffer_size){
                for(int i=0; i<input_channel[layer_id]; i++){
                    buffer[numElementsinBuffer] = 0;
                    numElementsinBuffer++;
                }
                for(int j=0; j<input_width[layer_id]*input_channel[layer_id]; j++){
                    buffer[numElementsinBuffer] = input.read();
                    numElementsinBuffer++;
                }
                for(int i=0; i<input_channel[layer_id]; i++){
                    buffer[numElementsinBuffer] = 0;
                    numElementsinBuffer++;
                }
            }
        }
    }

    for(int i =0; i<buffer_size; i++){
        cout << buffer[i] << " ";
    }
    cout << "numElementsinBuffer: " << numElementsinBuffer << endl;
    cout << endl;


    return 0;
}

#endif //NETWORK_H