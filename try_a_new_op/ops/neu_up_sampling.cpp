#include <cstdio>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
using namespace tensorflow;


/*
If want to set output to an arbitary number, use followed code in SetShapeFn:

        ::tensorflow::shape_inference::ShapeHandle dims1; //  a new shape object 
        c->WithRank(c->input(0), 3, &dims1); // the function of WithRank is to check whether the input is really a tensor with 3 dimension, if yes,  set the value of dims to the shape of input.
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 1), c->Dim(dims1, 0), c->Dim(dims1, 2)}); // use  the  value to build up a new shape.
                                                                                                                                                                                                                                                                            // this step can also be substitute by some known number e.g. {3,3,2}
        c->set_output(0, output);//set the output shape to calculated shape.
        return Status::OK();
*/

/*
this MACRO defined a operation called "NeuUpSampling".
And define the number. shape, name of input and output .
*/
REGISTER_OP("NeuUpSampling")
    .Input("origin: float32")
    .Input("idx_sparse: int32")
    .Input("feature: float32")
    .Output("upsampled: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));
        return Status::OK();
    });


REGISTER_OP("NeuGetIdxSparse")
    .Input("origin: float32")
    .Input("feature: float32")
    .Output("idx_sparse: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(c->input(0),0),c->Dim(c->input(0),1),c->Dim(c->input(0),2), 2}); 
        c->set_output(0, output);
        return Status::OK();
    });


REGISTER_OP("NeuUpSamplingGrad")
    .Input("origin: float32")
    .Input("idx_sparse: int32")
    .Input("feature: float32")
    .Input("upsampled_grad: float32")
    .Output("origin_grad: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


/*
return the poisition of  the max value in the cell

fisrt_index: global position x,y of first element of current cell
kernx: size of kern in x dim
kerny: size of kern in y dim
x: size of  tensor in x dim
y: size of tensor in y dim
data: pointer of first element of current cell
*/
std::vector<int> max_kernel( int kernx, int kerny, int x, int y, const float* data)
{
    int m_i = 0; // position in x
    int m_j = 0; // position in y
    std::vector<int> max_index(2);
    for(int i=0; i< kernx; ++i)
    {
        for(int j=0; j< kerny; ++j )
        {
            if(data[i*y*kerny+j]>data[m_i*y*kerny+m_j])
            {
                m_i = i;
                m_j = j;
            }
        }
    }
    max_index[0] = m_i;
    max_index[1] = m_j;
    return max_index;
}


 void neu_upsampling_cpu(int b, int x, int y, int kernx, int kerny, const float*  origin, const  int* idx_sparse, float* upsampled)
 {

     for(int batch=0; batch<b;++batch)
     {
         for(int i=0; i < x; ++i)
        {
            for(int j=0;j<y;++j)
            {
                upsampled[y*kerny*idx_sparse[batch*x*y*2 + (i*y+j)*2+0] + idx_sparse[batch*x*y*2 + (i*y+j)*2+1]] = origin[batch*x*y + i*y+j];
                upsampled += kerny;
            }
            upsampled += y*kerny*(kernx-1);
        }
     }
     
 }


void neu_get_idx_sparse_cpu(int b, int x, int y, int kernx, int kerny, int* idx_sparse, const float* feature)
{
    for(int batch=0; batch<b;++batch)
     {
         for(int i=0; i < x; ++i)
        {
            for(int j=0;j<y;++j)
            {
                std::vector<int>max_index = max_kernel(kernx, kerny, x, y, feature);
                idx_sparse[batch*x*y*2 + i*y *2+ j*2 + 0] = max_index[0];
                idx_sparse[batch*x*y*2 + i*y*2 + j*2 + 1] = max_index[1];
                feature += kerny;
            }
            feature += y*kerny*(kernx-1);
        }
     }
    
}


void neu_upsampling_grad_cpu(int b,int x,int y,int kernx,int kerny, const float *origin, const int* idx_sparse, const float* upsampled_grad, float* origin_grad)
{
    for(int batch=0; batch<b; ++batch)
    {
        for(int i=0; i<x;++i)
        {
            for(int j=0; j<y; ++j)
            {
                origin_grad[batch*x*y+ i*y+j] = upsampled_grad[idx_sparse[i*y *2+ j*2 + 0]*kerny*y + idx_sparse[i*y *2+ j*2 + 1]];
                upsampled_grad += kerny;
            }
            upsampled_grad += (kernx-1)*kerny*y;
        }
        idx_sparse += x*y*2;
    }
}


class NeuUpSamplingOp : public OpKernel {
    public:
        explicit NeuUpSamplingOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& origin_tensor = context->input(0);
            OP_REQUIRES(context, origin_tensor.dims()==3 , errors::InvalidArgument("NeuUpSampling expects origin with 3 dimension.")); // check wether the input satisfies  certain requirements.
                                                                                                                                                                                                                                                                                    // some useful method of  tensor for this checking are:
                                                                                                                                                                                                                                                                                    // awesome_tensor.shape()
                                                                                                                                                                                                                                                                                    // awesome_tensor.dim_size(n)
            int b = origin_tensor.shape().dim_size(0); // get batch size, i.e. the 1st dim of  input tensor
            int x = origin_tensor.shape().dim_size(1); // get row size
            int y = origin_tensor.shape().dim_size(2);// get column size

            const Tensor& idx_sparse_tensor = context->input(1);
            /*
            bool req1 = idx_sparse_tensor.shape().dim_size(1)%origin_tensor.shape().dim_size(1) ==0;
            bool req2 = idx_sparse_tensor.shape().dim_size(2)%origin_tensor.shape().dim_size(2) ==0;
            */
           bool req1 = (idx_sparse_tensor.shape().dim_size(0)==origin_tensor.shape().dim_size(0));
           bool req2 = (idx_sparse_tensor.shape().dim_size(1)==origin_tensor.shape().dim_size(1));
           bool req3 = (idx_sparse_tensor.shape().dim_size(2)==origin_tensor.shape().dim_size(2));
           bool req4 = (idx_sparse_tensor.shape().dim_size(3) == 2);
            OP_REQUIRES(context, idx_sparse_tensor.dims()==4 && req1 && req2 && req3 && req4, errors::InvalidArgument("NeuUpSampling expects a index, whose size should be the same as origin, and the last dim should be 2"));
            
            const Tensor& feature_tensor = context->input(2);

            bool req5 = feature_tensor.shape().dim_size(1)%origin_tensor.shape().dim_size(1) ==0;
            bool req6 = feature_tensor.shape().dim_size(2)%origin_tensor.shape().dim_size(2) ==0;
            OP_REQUIRES(context,  feature_tensor.dims()==3 && req5 && req6, errors::InvalidArgument("NeuUpSampling expects a feature with integer times size of  origin"));
            
            int kernx = feature_tensor.shape().dim_size(1)/origin_tensor.shape().dim_size(1);
            int kerny =  feature_tensor.shape().dim_size(2)/origin_tensor.shape().dim_size(2);

            Tensor *upsampled_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,x*kernx,y*kerny}, &upsampled_tensor));

            auto origin_flat = origin_tensor.flat<float>();
            const float *origin = &(origin_flat(0));
            auto idx_sparse_flat = idx_sparse_tensor.flat<int>();
            const int *idx_sparse = &(idx_sparse_flat(0));
            auto upsampled_flat = upsampled_tensor->flat<float>();
            float *upsampled = &(upsampled_flat(0));
            memset(upsampled, 0, sizeof(float)*b*x*kernx*y*kerny);
            neu_upsampling_cpu(b,x,y,kernx,kerny,origin,idx_sparse,upsampled);
        }
};
REGISTER_KERNEL_BUILDER(Name("NeuUpSampling").Device(DEVICE_CPU), NeuUpSamplingOp);


class NeuUpSamplingGradOp : public OpKernel {
    public:
        explicit NeuUpSamplingGradOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& origin_tensor = context->input(0);
            OP_REQUIRES(context, origin_tensor.dims()==3 , errors::InvalidArgument("NeuUpSampling expects origin with 3 dimension.")); // check wether the input satisfies  certain requirements.
                                                                                                                                                                                                                                                                                    // some useful method of  tensor for this checking are:
                                                                                                                                                                                                                                                                                    // awesome_tensor.shape()
                                                                                                                                                                                                                                                                                    // awesome_tensor.dim_size(n)
            int b = origin_tensor.shape().dim_size(0); // get batch size, i.e. the 1st dim of  input tensor
            int x = origin_tensor.shape().dim_size(1); // get row size
            int y = origin_tensor.shape().dim_size(2);// get column size

            const Tensor& idx_sparse_tensor = context->input(1);
            /*
            bool req1 = idx_sparse_tensor.shape().dim_size(1)%origin_tensor.shape().dim_size(1) ==0;
            bool req2 = idx_sparse_tensor.shape().dim_size(2)%origin_tensor.shape().dim_size(2) ==0;
            */
           bool req1 = (idx_sparse_tensor.shape().dim_size(0)==origin_tensor.shape().dim_size(0));
           bool req2 = (idx_sparse_tensor.shape().dim_size(1)==origin_tensor.shape().dim_size(1));
           bool req3 = (idx_sparse_tensor.shape().dim_size(2)==origin_tensor.shape().dim_size(2));
           bool req4 = (idx_sparse_tensor.shape().dim_size(3) == 2);
            OP_REQUIRES(context, idx_sparse_tensor.dims()==4 && req1 && req2 && req3 && req4, errors::InvalidArgument("NeuUpSampling expects a index, whose size should be the same as origin, and the last dim should be 2"));
            
            const Tensor& feature_tensor = context->input(2);

            bool req5 = feature_tensor.shape().dim_size(1)%origin_tensor.shape().dim_size(1) ==0;
            bool req6 = feature_tensor.shape().dim_size(2)%origin_tensor.shape().dim_size(2) ==0;
            OP_REQUIRES(context,  feature_tensor.dims()==3 && req5 && req6, errors::InvalidArgument("NeuUpSampling expects a feature with integer times size of  origin"));
            
            int kernx = feature_tensor.shape().dim_size(1)/origin_tensor.shape().dim_size(1);
            int kerny =  feature_tensor.shape().dim_size(2)/origin_tensor.shape().dim_size(2);

            const Tensor& upsampled_grad_tensor = context->input(3);
            OP_REQUIRES(context, upsampled_grad_tensor.dims()==3 , errors::InvalidArgument("NeuUpSampling expects origin with 3 dimension.")); 

            Tensor *origin_grad_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,x,y}, &origin_grad_tensor));

            auto origin_flat = origin_tensor.flat<float>();
            const float *origin = &(origin_flat(0));
            auto idx_sparse_flat = idx_sparse_tensor.flat<int>();
            const int *idx_sparse = &(idx_sparse_flat(0));
            auto upsampled_grad_flat = upsampled_grad_tensor.flat<float>();
            const float *upsampled_grad = &(upsampled_grad_flat(0));
            auto origin_grad_flat = origin_grad_tensor->flat<float>();
            float *origin_grad = &(origin_grad_flat(0));
            memset(origin_grad, 0, sizeof(float)*b*x*y);
            neu_upsampling_grad_cpu(b,x,y,kernx,kerny,origin,idx_sparse,upsampled_grad,origin_grad);
        }
};
REGISTER_KERNEL_BUILDER(Name("NeuUpSamplingGrad").Device(DEVICE_CPU), NeuUpSamplingGradOp);


class NeuGetIdxSparseOp : public OpKernel {
    public:
        explicit NeuGetIdxSparseOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& origin_tensor = context->input(0);
            OP_REQUIRES(context, origin_tensor.dims()==3 , errors::InvalidArgument("NeuUpSampling expects origin with 3 dimension.")); // check wether the input satisfies  certain requirements.
                                                                                                                                                                                                                                                                                    // some useful method of  tensor for this checking are:
                                                                                                                                                                                                                                                                                    // awesome_tensor.shape()
                                                                                                                                                                                                                                                                                    // awesome_tensor.dim_size(n)
            int b = origin_tensor.shape().dim_size(0); // get batch size, i.e. the 1st dim of  input tensor
            int x = origin_tensor.shape().dim_size(1); // get row size
            int y = origin_tensor.shape().dim_size(2);// get column size

            const Tensor& feature_tensor = context->input(1);
            bool req1 = feature_tensor.shape().dim_size(1)%origin_tensor.shape().dim_size(1) ==0;
            bool req2 = feature_tensor.shape().dim_size(2)%origin_tensor.shape().dim_size(2) ==0;
            OP_REQUIRES(context, feature_tensor.dims()==3 && req1 && req2, errors::InvalidArgument("NeuUpSampling expects a index, which size with integer times of row and col w.r.t. origin"));

            int kernx = feature_tensor.shape().dim_size(1)/origin_tensor.shape().dim_size(1);
            int kerny =  feature_tensor.shape().dim_size(2)/origin_tensor.shape().dim_size(2);

            Tensor *idx_sparse_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,x,y,2}, &idx_sparse_tensor));

            auto origin_flat = origin_tensor.flat<float>();
            const float *origin = &(origin_flat(0));
            auto feature_flat = feature_tensor.flat<float>();
            const float *feature = &(feature_flat(0));
            auto idx_sparse_flat = idx_sparse_tensor->flat<int>();
            int *idx_sparse = &(idx_sparse_flat(0));
            neu_get_idx_sparse_cpu(b,x,y,kernx,kerny,idx_sparse,feature);
        }
};
REGISTER_KERNEL_BUILDER(Name("NeuGetIdxSparse").Device(DEVICE_CPU), NeuGetIdxSparseOp);